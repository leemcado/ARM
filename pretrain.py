from typing import Optional, Any, Sequence, List
from dataclasses import dataclass, field
import os
import math
import yaml
import shutil
from collections import deque

import torch
import torch.distributed as dist
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import tqdm
import wandb
import coolname
import hydra
import pydantic
from omegaconf import DictConfig
from adam_atan2 import AdamATan2

from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig, PuzzleDatasetMetadata
from utils.functions import load_model_class, get_model_source_path
from models.sparse_embedding import CastedSparseEmbeddingSignSGD_Distributed


class LossConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')
    
    name: str


class ArchConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')

    name: str
    loss: LossConfig


class PretrainConfig(pydantic.BaseModel):
    # Config
    arch: ArchConfig
    # Data
    data_path: str

    # Hyperparams
    global_batch_size: int
    epochs: int

    lr: float
    lr_min_ratio: float
    lr_warmup_steps: int

    weight_decay: float
    beta1: float
    beta2: float

    # Puzzle embedding
    puzzle_emb_lr: float
    puzzle_emb_weight_decay: float

    # Names
    project_name: Optional[str] = None
    run_name: Optional[str] = None
    checkpoint_path: Optional[str] = None

    # Extras
    seed: int = 0
    checkpoint_every_eval: bool = False
    eval_interval: Optional[int] = None
    eval_save_outputs: List[str] = []


@dataclass
class TrainState:
    model: nn.Module
    optimizers: Sequence[torch.optim.Optimizer]
    optimizer_lrs: Sequence[float]
    carry: Any

    step: int
    total_steps: int
    
    # ARM을 위한 추가 상태
    routing_history: deque = field(default_factory=deque)


def create_dataloader(config: PretrainConfig, split: str, rank: int, world_size: int, **kwargs):
    dataset = PuzzleDataset(PuzzleDatasetConfig(
        seed=config.seed,
        dataset_path=config.data_path,
        rank=rank,
        num_replicas=world_size,
        **kwargs
    ), split=split)
    dataloader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=1,
        prefetch_factor=8,
        pin_memory=True,
        persistent_workers=True
    )
    return dataloader, dataset.metadata

def create_model(config: PretrainConfig, train_metadata: PuzzleDatasetMetadata, world_size: int):
    model_cfg = dict(
        **config.arch.__pydantic_extra__,
        batch_size=config.global_batch_size // world_size,
        vocab_size=train_metadata.vocab_size,
        seq_len=train_metadata.seq_len,
        num_puzzle_identifiers=train_metadata.num_puzzle_identifiers,
        causal=False
    )

    model_cls = load_model_class(config.arch.name)
    loss_head_cls = load_model_class(config.arch.loss.name)

    with torch.device("cuda"):
        model: nn.Module = model_cls(model_cfg)
        model = loss_head_cls(model, **config.arch.loss.__pydantic_extra__)
        if "DISABLE_COMPILE" not in os.environ:
            model = torch.compile(model, dynamic=False)

        if world_size > 1:
            with torch.no_grad():
                for param in list(model.parameters()) + list(model.buffers()):
                    dist.broadcast(param, src=0)

    # --- 옵티마이저 설정 수정 ---
    # 이제 옵티마이저는 AdamATan2 하나만 존재합니다.
    optimizers = [
        AdamATan2(
            model.parameters(),
            lr=0,
            weight_decay=config.weight_decay,
            betas=(config.beta1, config.beta2)
        )
    ]
    # 학습률 설정도 하나로 통일합니다.
    optimizer_lrs = [
        config.lr
    ]

    return model, optimizers, optimizer_lrs



def cosine_schedule_with_warmup_lr_lambda(
    current_step: int, *, base_lr: float, num_warmup_steps: int, num_training_steps: int, min_ratio: float = 0.0, num_cycles: float = 0.5
):
    if current_step < num_warmup_steps:
        return base_lr * float(current_step) / float(max(1, num_warmup_steps))

    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return base_lr * (min_ratio + max(0.0, (1 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))))


def init_train_state(config: PretrainConfig, train_metadata: PuzzleDatasetMetadata, world_size: int):
    # Estimated total training steps
    total_steps = int(config.epochs * train_metadata.total_groups * train_metadata.mean_puzzle_examples / config.global_batch_size)

    # Model
    model, optimizers, optimizer_lrs = create_model(config, train_metadata, world_size=world_size)
    
    # ARM의 수렴 윈도우 크기를 설정 파일에서 가져옵니다.
    convergence_window = config.arch.__pydantic_extra__.get('convergence_window', 1000)

    return TrainState(
        step=0,
        total_steps=total_steps,

        model=model,
        optimizers=optimizers,
        optimizer_lrs=optimizer_lrs,
        carry=None,
        routing_history=deque(maxlen=convergence_window)
    )


def save_train_state(config: PretrainConfig, train_state: TrainState):
    # FIXME: Only saved model.
    if config.checkpoint_path is None:
        return

    os.makedirs(config.checkpoint_path, exist_ok=True)
    torch.save(train_state.model.state_dict(), os.path.join(config.checkpoint_path, f"step_{train_state.step}"))


def compute_lr(base_lr: float, config: PretrainConfig, train_state: TrainState):
    return cosine_schedule_with_warmup_lr_lambda(
        current_step=train_state.step,
        base_lr=base_lr,
        num_warmup_steps=round(config.lr_warmup_steps),
        num_training_steps=train_state.total_steps,
        min_ratio=config.lr_min_ratio
    )

# +++ ARM을 위한 새로운 train_batch 함수 +++
def train_batch(config: PretrainConfig, train_state: TrainState, batch: Any, global_batch_size: int, rank: int, world_size: int):
    train_state.step += 1
    if train_state.step > train_state.total_steps:  # At most train_total_steps
        return

    arch_config = config.arch.__pydantic_extra__
    
    # --- A. 모듈 성장 단계 ---
    if (train_state.step > arch_config['convergence_window'] and 
        train_state.step % arch_config['convergence_window'] == 0):
        
        # 라우팅 엔트로피 계산 (분모가 0이 되는 것 방지)
        history_tensor = torch.tensor(list(train_state.routing_history), dtype=torch.long)
        if len(history_tensor) > 0:
            counts = torch.bincount(history_tensor, minlength=len(train_state.model.model.inner.reasoning_modules))
            if counts.sum() > 0:
                probs = counts.float() / counts.sum()
                entropy = -torch.sum(probs * torch.log(probs + 1e-9)).item()

                if rank == 0:
                    wandb.log({"train/routing_entropy": entropy}, step=train_state.step)

                if entropy < arch_config['entropy_threshold']:
                    # 새 모듈 추가 (모든 랭크에서 동일하게 실행되어야 함)
                    if train_state.model.model.inner.add_new_expert_module():
                        # 새 z_state를 carry에 추가 (모든 랭크에서)
                        batch_size = batch['inputs'].shape[0]
                        new_z = torch.zeros(batch_size, config.arch.seq_len, arch_config['hidden_size'], dtype=getattr(torch, arch_config['forward_dtype']), device="cuda")
                        train_state.carry.inner_carry.z_states.append(new_z)
                        if rank == 0:
                            print(f"Step {train_state.step}: Model size increased to {len(train_state.model.model.inner.reasoning_modules)} modules.")

    # --- B. 추론 및 학습 단계 ---
    batch = {k: v.cuda() for k, v in batch.items()}
    if train_state.carry is None:
        with torch.device("cuda"):
            train_state.carry = train_state.model.initial_carry(batch)  # type: ignore

    # ACT 로직과 결합하여 carry 객체 업데이트
    train_state.carry.inner_carry = train_state.model.model.inner.reset_carry(train_state.carry.halted, train_state.carry.inner_carry)
    train_state.carry.steps = torch.where(train_state.carry.halted, 0, train_state.carry.steps)
    train_state.carry.current_data = {k: torch.where(train_state.carry.halted.view((-1, ) + (1, ) * (batch[k].ndim - 1)), batch[k], v) for k, v in train_state.carry.current_data.items()}

    # 1. 모듈 선택 (시상)
    num_modules = len(train_state.model.model.inner.reasoning_modules)
    input_embeddings = train_state.model.model.inner._input_embeddings(train_state.carry.current_data["inputs"], train_state.carry.current_data["puzzle_identifiers"])
    
    with torch.no_grad():
        # 시상 모듈 입력으로는 시퀀스의 첫 번째 토큰 임베딩을 사용 (CLS 토큰처럼)
        predicted_errors = train_state.model.model.inner.thalamus_gate(input_embeddings[:, 0, :])[:, :num_modules]
        chosen_module_idx = torch.argmin(predicted_errors, dim=1)[0].item() # Assume batch is uniform
        train_state.routing_history.append(chosen_module_idx)

    # 2. z_reason 계산
    z_states = train_state.carry.inner_carry.z_states
    other_z_states = [z for i, z in enumerate(z_states) if i != chosen_module_idx]
    other_z_sum = torch.stack(other_z_states, dim=0).sum(dim=0) if len(other_z_states) > 0 else 0
    
    inactive_contrib = input_embeddings * (arch_config['max_modules'] - num_modules)
    z_reason = other_z_sum + inactive_contrib

    # 3 & 4. 1-Step Backpropagation
    seq_info = dict(cos_sin=train_state.model.model.inner.rotary_emb() if hasattr(train_state.model.model.inner, "rotary_emb") else None)
    
    # no_grad 계산
    with torch.no_grad():
        z_active_nograd = z_states[chosen_module_idx].clone()
        for _ in range(arch_config['inner_loops']):
            module_input = z_active_nograd + z_reason
            z_active_nograd = train_state.model.model.inner.reasoning_modules[chosen_module_idx](module_input, **seq_info)
        
        final_active_signal = z_active_nograd
        z_others_nograd = []
        for i in range(num_modules):
            if i != chosen_module_idx:
                module_input = z_states[i] + final_active_signal
                z_others_nograd.append(train_state.model.model.inner.reasoning_modules[i](module_input, **seq_info))

    # grad 계산
    z_active_grad = z_states[chosen_module_idx]
    for _ in range(arch_config['inner_loops']):
        module_input_grad = z_active_grad + z_reason
        z_active_grad = train_state.model.model.inner.reasoning_modules[chosen_module_idx](module_input_grad, **seq_info)

    final_active_signal_grad = z_active_grad
    z_others_grad = []
    for i in range(num_modules):
        if i != chosen_module_idx:
            module_input_grad = z_states[i] + final_active_signal_grad
            z_others_grad.append(train_state.model.model.inner.reasoning_modules[i](module_input_grad, **seq_info))

    # 5. 손실 계산
    final_z_states_grad = [final_active_signal_grad] + z_others_grad
    final_logits = [train_state.model.model.inner.lm_head(z[:, -config.arch.seq_len:, :]) for z in final_z_states_grad]
    
    q_logits = train_state.model.model.inner.q_head(final_active_signal_grad[:, 0, :]).to(torch.float32)
    q_halt_logits, q_continue_logits = q_logits[..., 0], q_logits[..., 1]
    
    # ACT 정지 결정 및 다음 스텝을 위한 carry 업데이트
    train_state.carry.steps += 1
    is_last_step = train_state.carry.steps >= train_state.model.model.config.halt_max_steps
    
    # ACT halt 로직 (HRM 원본과 유사)
    with torch.no_grad():
        halted = is_last_step
        if train_state.training and (train_state.model.model.config.halt_max_steps > 1):
            halted = halted | (q_halt_logits > q_continue_logits)
            min_halt_steps = (torch.rand_like(q_halt_logits) < train_state.model.model.config.halt_exploration_prob) * torch.randint_like(train_state.carry.steps, low=2, high=train_state.model.model.config.halt_max_steps + 1)
            halted = halted & (train_state.carry.steps >= min_halt_steps)
    train_state.carry.halted = halted

    # target_q 계산
    target_q_continue = None
    if train_state.training:
        with torch.no_grad():
            next_q_logits = train_state.model.model.inner.q_head(final_active_signal.detach()[:, 0, :])
            next_q_halt, next_q_continue = next_q_logits[..., 0], next_q_logits[..., 1]
            target_q_continue = torch.sigmoid(torch.where(is_last_step, next_q_halt, torch.maximum(next_q_halt, next_q_continue)))

    # Loss Head 호출
    _, lm_q_loss, metrics, _, _ = train_state.model(
        carry=train_state.carry, batch=batch, final_logits=final_logits,
        q_halt_logits=q_halt_logits, q_continue_logits=q_continue_logits,
        target_q_continue=target_q_continue
    )
    
    # Gating Loss 계산
    predicted_error_for_chosen = predicted_errors[:, chosen_module_idx]
    lm_loss = metrics.get('lm_loss', torch.tensor(0.0, device="cuda"))
    gating_loss = F.mse_loss(predicted_error_for_chosen, lm_loss.detach())
    
    total_loss = lm_q_loss + gating_loss
    
    # 역전파
    ((1 / global_batch_size) * total_loss).backward()
    
    # Allreduce
    if world_size > 1:
        for param in train_state.model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad)
            
    # 옵티마이저 스텝
    lr_this_step = None    
    for optim, base_lr in zip(train_state.optimizers, train_state.optimizer_lrs):
        lr_this_step = compute_lr(base_lr, config, train_state)
        for param_group in optim.param_groups:
            param_group['lr'] = lr_this_step
        optim.step()
        optim.zero_grad()

    # 다음 스텝을 위해 z_states 업데이트
    with torch.no_grad():
        train_state.carry.inner_carry.z_states[chosen_module_idx].copy_(final_active_signal)
        other_idx_counter = 0
        for i in range(num_modules):
            if i != chosen_module_idx:
                train_state.carry.inner_carry.z_states[i].copy_(z_others_nograd[other_idx_counter])
                other_idx_counter += 1
    
    # 메트릭 로깅
    if len(metrics):
        metric_keys = list(sorted(metrics.keys()))
        metric_values = torch.stack([metrics[k] for k in metric_keys])
        if world_size > 1:
            dist.reduce(metric_values, dst=0)

        if rank == 0:
            metric_values = metric_values.cpu().numpy()
            reduced_metrics = {k: metric_values[i] for i, k in enumerate(metric_keys)}
            
            count = max(reduced_metrics["count"], 1)
            reduced_metrics = {f"train/{k}": v / (global_batch_size if k.endswith("loss") else count) for k, v in reduced_metrics.items()}
            reduced_metrics["train/gating_loss"] = gating_loss.item() / global_batch_size
            reduced_metrics["train/lr"] = lr_this_step
            return reduced_metrics


def evaluate(config: PretrainConfig, train_state: TrainState, eval_loader: torch.utils.data.DataLoader, eval_metadata: PuzzleDatasetMetadata, rank: int, world_size: int):
    with torch.inference_mode():
        set_ids = {k: idx for idx, k in enumerate(eval_metadata.sets)}
        
        all_preds = {}
        arch_config = config.arch.__pydantic_extra__

        metric_keys = []
        metric_values_list = []
        
        for set_name, batch, global_batch_size in eval_loader:
            batch = {k: v.cuda() for k, v in batch.items()}
            with torch.device("cuda"):
                carry = train_state.model.initial_carry(batch)

            # 평가 시에는 max_steps까지 항상 실행
            for step in range(arch_config['halt_max_steps']):
                carry.inner_carry = train_state.model.model.inner.reset_carry(carry.halted, carry.inner_carry)
                carry.steps = torch.where(carry.halted, 0, carry.steps)
                carry.current_data = {k: torch.where(carry.halted.view((-1, ) + (1, ) * (batch[k].ndim - 1)), batch[k], v) for k, v in carry.current_data.items()}

                num_modules = len(train_state.model.model.inner.reasoning_modules)
                input_embeddings = train_state.model.model.inner._input_embeddings(carry.current_data["inputs"], carry.current_data["puzzle_identifiers"])
                
                predicted_errors = train_state.model.model.inner.thalamus_gate(input_embeddings[:, 0, :])[:, :num_modules]
                chosen_module_idx = torch.argmin(predicted_errors, dim=1)[0].item()

                z_states = carry.inner_carry.z_states
                other_z_states = [z for i, z in enumerate(z_states) if i != chosen_module_idx]
                other_z_sum = torch.stack(other_z_states, dim=0).sum(dim=0) if len(other_z_states) > 0 else 0
                inactive_contrib = input_embeddings * (arch_config['max_modules'] - num_modules)
                z_reason = other_z_sum + inactive_contrib

                seq_info = dict(cos_sin=train_state.model.model.inner.rotary_emb() if hasattr(train_state.model.model.inner, "rotary_emb") else None)

                z_active_next = z_states[chosen_module_idx]
                for _ in range(arch_config['inner_loops']):
                    module_input = z_active_next + z_reason
                    z_active_next = train_state.model.model.inner.reasoning_modules[chosen_module_idx](module_input, **seq_info)
                
                final_active_signal = z_active_next
                z_others_next = []
                for i in range(num_modules):
                    if i != chosen_module_idx:
                        module_input = z_states[i] + final_active_signal
                        z_others_next.append(train_state.model.model.inner.reasoning_modules[i](module_input, **seq_info))

                carry.steps += 1
                carry.halted = (carry.steps >= arch_config['halt_max_steps'])

                carry.inner_carry.z_states[chosen_module_idx].copy_(final_active_signal)
                other_idx_counter = 0
                for i in range(num_modules):
                    if i != chosen_module_idx:
                        carry.inner_carry.z_states[i].copy_(z_others_next[other_idx_counter])
                        other_idx_counter += 1

                if carry.halted.all():
                    break
            
            # 최종 결과로 메트릭 계산
            puzzle_emb_len = arch_config.get('puzzle_emb_len', -(arch_config.get('puzzle_emb_ndim', 0) // -arch_config['hidden_size']))
            final_logits = train_state.model.model.inner.lm_head(final_active_signal[:, puzzle_emb_len:, :])
            labels = carry.current_data["labels"]
            
            mask = labels != IGNORE_LABEL_ID
            loss_counts = mask.sum(-1)
            
            is_correct = mask & (torch.argmax(final_logits, dim=-1) == labels)
            seq_is_correct = is_correct.sum(-1) == loss_counts
            
            metrics = {
                "count": torch.tensor(loss_counts.shape[0], device="cuda"),
                "accuracy": (is_correct.float().sum() / loss_counts.sum().clamp_min(1)),
                "exact_accuracy": seq_is_correct.sum()
            }
            if not metric_keys:
                metric_keys = sorted(metrics.keys())
            metric_values_list.append(torch.stack([metrics[k] for k in metric_keys]))

            # 결과 저장
            if len(config.eval_save_outputs):
                preds = {"logits": final_logits}
                for collection in (batch, preds):
                    for k, v in collection.items():
                        if k in config.eval_save_outputs:
                            all_preds.setdefault(k, [])
                            all_preds[k].append(v.cpu())
        
        if len(all_preds):
            all_preds = {k: torch.cat(v, dim=0) for k, v in all_preds.items()}
            os.makedirs(config.checkpoint_path, exist_ok=True)
            torch.save(all_preds, os.path.join(config.checkpoint_path, f"step_{train_state.step}_all_preds.{rank}.pt"))

        if metric_values_list:
            metric_values = torch.stack(metric_values_list).sum(0)
            if world_size > 1:
                dist.reduce(metric_values, dst=0)
            
            if rank == 0:
                metric_values = metric_values.cpu().numpy()
                reduced_metrics = {k: metric_values[i] for i, k in enumerate(metric_keys)}
                count = reduced_metrics.pop("count")
                reduced_metrics = {f"eval/{k}": v / count for k, v in reduced_metrics.items()}
                return reduced_metrics


def save_code_and_config(config: PretrainConfig):
    if config.checkpoint_path is None or wandb.run is None:
        return

    os.makedirs(config.checkpoint_path, exist_ok=True)
    
    # ARM 모델 경로 추가
    code_list = [
        get_model_source_path(config.arch.name),
        get_model_source_path(config.arch.loss.name)
    ]
    for code_file in code_list:
        if code_file is not None:
            code_name = os.path.basename(code_file)
            shutil.copy(code_file, os.path.join(config.checkpoint_path, code_name))

    config_file = os.path.join(config.checkpoint_path, "all_config.yaml")
    with open(config_file, "wt") as f:
        yaml.dump(config.model_dump(), f)

    wandb.run.log_code(config.checkpoint_path)


def load_synced_config(hydra_config: DictConfig, rank: int, world_size: int) -> PretrainConfig:
    objects = [None]
    if rank == 0:
        config = PretrainConfig(**OmegaConf.to_container(hydra_config)) # type: ignore

        if config.project_name is None:
            config.project_name = f"{os.path.basename(config.data_path).capitalize()} ARM-torch"
        if config.run_name is None:
            config.run_name = f"{config.arch.name.split('@')[-1]} {coolname.generate_slug(2)}"
        if config.checkpoint_path is None:
            config.checkpoint_path = os.path.join("checkpoints", config.project_name, config.run_name)

        objects = [config]

    if world_size > 1:
        dist.broadcast_object_list(objects, src=0)

    return objects[0]


@hydra.main(config_path="config", config_name="cfg_pretrain", version_base=None)
def launch(hydra_config: DictConfig):
    RANK = 0
    WORLD_SIZE = 1

    if "LOCAL_RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        RANK = dist.get_rank()
        WORLD_SIZE = dist.get_world_size()
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        
    config = load_synced_config(hydra_config, rank=RANK, world_size=WORLD_SIZE)

    torch.random.manual_seed(config.seed + RANK)

    train_epochs_per_iter = config.eval_interval if config.eval_interval is not None else config.epochs
    total_iters = config.epochs // train_epochs_per_iter
    assert config.epochs % train_epochs_per_iter == 0, "Eval interval must be a divisor of total epochs."

    train_loader, train_metadata = create_dataloader(config, "train", test_set_mode=False, epochs_per_iter=train_epochs_per_iter, global_batch_size=config.global_batch_size, rank=RANK, world_size=WORLD_SIZE)
    eval_loader,  eval_metadata  = create_dataloader(config, "test", test_set_mode=True, epochs_per_iter=1, global_batch_size=config.global_batch_size, rank=RANK, world_size=WORLD_SIZE)

    train_state = init_train_state(config, train_metadata, world_size=WORLD_SIZE)

    progress_bar = None
    if RANK == 0:
        progress_bar = tqdm.tqdm(total=train_state.total_steps)
        wandb.init(project=config.project_name, name=config.run_name, config=config.model_dump(), settings=wandb.Settings(_disable_stats=True))
        wandb.log({"num_params": sum(x.numel() for x in train_state.model.parameters())}, step=0)
        save_code_and_config(config)

    for _iter_id in range(total_iters):
        print (f"[Rank {RANK}, World Size {WORLD_SIZE}]: Epoch {_iter_id * train_epochs_per_iter}")

        train_state.model.train()
        for set_name, batch, global_batch_size_effective in train_loader:
            metrics = train_batch(config, train_state, batch, global_batch_size_effective, rank=RANK, world_size=WORLD_SIZE)

            if RANK == 0 and metrics is not None:
                wandb.log(metrics, step=train_state.step)
                progress_bar.update(train_state.step - progress_bar.n)

        train_state.model.eval()
        eval_metrics = evaluate(config, train_state, eval_loader, eval_metadata, rank=RANK, world_size=WORLD_SIZE)

        if RANK == 0 and eval_metrics is not None:
            wandb.log(eval_metrics, step=train_state.step)
            
        if RANK == 0 and (config.checkpoint_every_eval or (_iter_id == total_iters - 1)):
            save_train_state(config, train_state)

    if dist.is_initialized():
        dist.destroy_process_group()
    if wandb.run:
        wandb.finish()


if __name__ == "__main__":
    launch()