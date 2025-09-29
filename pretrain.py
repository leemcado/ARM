from typing import Optional, Any, Sequence, List
from dataclasses import dataclass, field
import os
import math
import yaml
import shutil
from collections import deque
from omegaconf import DictConfig, OmegaConf
import torch
import torch.distributed as dist
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import tqdm
import wandb
import coolname
import hydra
import pydantic
from omegaconf import DictConfig
from adam_atan2 import AdamATan2
from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig, PuzzleDatasetMetadata
from utils.functions import load_model_class, get_model_source_path
from models.losses import IGNORE_LABEL_ID
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
    # ARM 성장 메커니즘을 위한 상태 변수들
    gating_grad_variances: deque = field(default_factory=deque)
    min_predicted_errors: deque = field(default_factory=deque)
    hard_problem_threshold: float = float('inf')
    system_converged: bool = False
    is_in_forced_allocation_phase: bool = False
    forced_allocation_steps_left: int = 0
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
        num_puzzle_identifiers=train_metadata.num_puzzle_identifiers
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
    # 퍼즐 임베딩 옵티마이저 제거 후 AdamATan2만 사용
    optimizers = [
        AdamATan2(
            model.parameters(),
            lr=0,
            weight_decay=config.weight_decay,
            betas=(config.beta1, config.beta2)
        )
    ]
    optimizer_lrs = [config.lr]
    return model, optimizers, optimizer_lrs
def cosine_schedule_with_warmup_lr_lambda(
    current_step: int, *, base_lr: float, num_warmup_steps: int, num_training_steps: int, min_ratio: float = 0.0, num_cycles: float = 0.5
):
    if current_step < num_warmup_steps:
        return base_lr * float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return base_lr * (min_ratio + max(0.0, (1 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))))
def init_train_state(config: PretrainConfig, train_metadata: PuzzleDatasetMetadata, world_size: int):
    total_steps = int(config.epochs * train_metadata.total_groups * train_metadata.mean_puzzle_examples / config.global_batch_size)
    model, optimizers, optimizer_lrs = create_model(config, train_metadata, world_size=world_size)
    convergence_interval = config.arch.__pydantic_extra__.get('convergence_check_interval', 1000)
    return TrainState(
        step=0,
        total_steps=total_steps,
        model=model,
        optimizers=optimizers,
        optimizer_lrs=optimizer_lrs,
        carry=None,
        gating_grad_variances=deque(maxlen=convergence_interval),
        min_predicted_errors=deque(maxlen=convergence_interval)
    )
def save_train_state(config: PretrainConfig, train_state: TrainState):
    if config.checkpoint_path is None:
        return
    os.makedirs(config.checkpoint_path, exist_ok=True)
    torch.save(train_state.model.state_dict(), os.path.join(config.checkpoint_path, f"step_{train_state.step}.pth"))
def compute_lr(base_lr: float, config: PretrainConfig, train_state: TrainState):
    return cosine_schedule_with_warmup_lr_lambda(
        current_step=train_state.step,
        base_lr=base_lr,
        num_warmup_steps=round(config.lr_warmup_steps),
        num_training_steps=train_state.total_steps,
        min_ratio=config.lr_min_ratio
    )
def train_batch(config: PretrainConfig, train_state: TrainState, batch: Any, global_batch_size: int, rank: int, world_size: int):
    train_state.step += 1
    if train_state.step > train_state.total_steps:
        return

    arch_config = config.arch.__pydantic_extra__
    batch = {k: v.cuda() for k, v in batch.items()}
    if train_state.carry is None:
        with torch.device("cuda"):
            train_state.carry = train_state.model.initial_carry(batch)

    # ACT 로직과 결합하여 carry 객체 업데이트
    train_state.carry.inner_carry = train_state.model.model.inner.reset_carry(train_state.carry.halted, train_state.carry.inner_carry)
    train_state.carry.steps = torch.where(train_state.carry.halted, 0, train_state.carry.steps)
    train_state.carry.current_data = {k: torch.where(train_state.carry.halted.view((-1, ) + (1, ) * (batch[k].ndim - 1)), batch[k], v) for k, v in train_state.carry.current_data.items()}

    # 순방향 계산 준비
    num_modules = len(train_state.model.model.inner.reasoning_modules)
    input_embeddings = train_state.model.model.inner._input_embeddings(train_state.carry.current_data["inputs"], train_state.carry.current_data.get("puzzle_identifiers"))

    # '시상 게이트'의 예측은 그래디언트 추적이 필요함 (학습을 위해)
    predicted_errors = train_state.model.model.inner.thalamus_gate(input_embeddings[:, 0, :])[:, :num_modules]

    # 'argmin'을 통한 모듈 선택은 그래디언트가 필요 없음
    with torch.no_grad():
        chosen_module_idx = torch.argmin(predicted_errors, dim=1)[0].item()

    # --- 강제 할당 단계인 경우, 라우팅 결정 덮어쓰기 ---
    if train_state.is_in_forced_allocation_phase:
        if num_modules > 1:
            min_error_among_old_modules_per_sample = torch.min(predicted_errors[:, :-1], dim=1).values
            max_of_min_errors = min_error_among_old_modules_per_sample.max().item()
            if max_of_min_errors > train_state.hard_problem_threshold:
                chosen_module_idx = num_modules - 1

    z_states = train_state.carry.inner_carry.z_states
    other_z_states = [z for i, z in enumerate(z_states) if i != chosen_module_idx]
    other_z_sum = torch.stack(other_z_states, dim=0).sum(dim=0) if len(other_z_states) > 0 else 0
    z_active = z_states[chosen_module_idx]
    inactive_contrib = z_active * (arch_config['max_modules'] - num_modules)
    z_reason = other_z_sum + inactive_contrib

    seq_info = dict(cos_sin=train_state.model.model.inner.rotary_emb() if hasattr(train_state.model.model.inner, "rotary_emb") else None)
    
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

    final_z_states_grad = [final_active_signal_grad] + z_others_grad
    seq_len = train_state.model.model.config.seq_len
    final_logits = [train_state.model.model.inner.lm_head(z[:, -seq_len:, :]) for z in final_z_states_grad]
    
    q_logits = train_state.model.model.inner.q_head(final_active_signal_grad[:, 0, :]).to(torch.float32)
    q_halt_logits, q_continue_logits = q_logits[..., 0], q_logits[..., 1]
    
    train_state.carry.steps += 1
    is_last_step = train_state.carry.steps >= train_state.model.model.config.halt_max_steps
    
    with torch.no_grad():
        halted = is_last_step
        if train_state.model.training and (train_state.model.model.config.halt_max_steps > 1):
                halted = halted | (q_halt_logits > q_continue_logits)
                min_halt_steps = (torch.rand_like(q_halt_logits) < train_state.model.model.config.halt_exploration_prob) * torch.randint_like(train_state.carry.steps, low=2, high=train_state.model.model.config.halt_max_steps + 1)
                halted = halted & (train_state.carry.steps >= min_halt_steps)
    train_state.carry.halted = halted

    target_q_continue = None
    if train_state.model.training:
        with torch.no_grad():
            next_q_logits = train_state.model.model.inner.q_head(final_active_signal.detach()[:, 0, :])
            next_q_halt, next_q_continue = next_q_logits[..., 0], next_q_logits[..., 1]
            target_q_continue = torch.sigmoid(torch.where(is_last_step, next_q_halt, torch.maximum(next_q_halt, next_q_continue)))

    _, lm_q_loss, metrics, _, _ = train_state.model(
        carry=train_state.carry, batch=batch, final_logits=final_logits,
        q_halt_logits=q_halt_logits, q_continue_logits=q_continue_logits,
        target_q_continue=target_q_continue
    )
    
    predicted_error_for_chosen = predicted_errors[:, chosen_module_idx]
    lm_loss = metrics.get('lm_loss', torch.tensor(0.0, device="cuda"))
    
    # --- 최종 수정: 데이터 타입 일치 ---
    # lm_loss를 predicted_error_for_chosen의 데이터 타입(BFloat16)으로 변환
    gating_loss = F.mse_loss(predicted_error_for_chosen, lm_loss.detach().to(predicted_error_for_chosen.dtype))
    
    total_loss = lm_q_loss + gating_loss
    
    total_loss.backward()

    # --- 역전파 직후, 옵티마이저 실행 전 ---
    # A. 상태 모니터링
    all_grads = []
    for param in train_state.model.model.inner.thalamus_gate.parameters():
        if param.grad is not None:
            all_grads.append(param.grad.view(-1))
    
    current_grad_variance = torch.var(torch.cat(all_grads)).item() if len(all_grads) > 0 else 0.0
    train_state.gating_grad_variances.append(current_grad_variance)
    min_error_this_step = torch.min(predicted_errors).item()
    train_state.min_predicted_errors.append(min_error_this_step)
    
    # B. 모듈 성장 상태 머신
    if train_state.is_in_forced_allocation_phase:
        train_state.forced_allocation_steps_left -= 1
        if train_state.forced_allocation_steps_left <= 0:
            train_state.is_in_forced_allocation_phase = False
            train_state.system_converged = False
    elif train_state.system_converged:
        if len(train_state.min_predicted_errors) > 0:
            errors_np = np.array(list(train_state.min_predicted_errors))
            train_state.hard_problem_threshold = np.percentile(errors_np, (1 - arch_config['rate_hardprob']) * 100)
        
        if train_state.model.model.inner.add_new_expert_module():
            train_state.is_in_forced_allocation_phase = True
            train_state.forced_allocation_steps_left = arch_config['forced_allocation_duration']
            # 새 z_state를 carry에 추가
            batch_size = batch['inputs'].shape[0]
            new_z = torch.zeros(batch_size, train_state.model.model.config.seq_len, arch_config['hidden_size'], dtype=getattr(torch, train_state.model.model.config.forward_dtype), device="cuda")
            train_state.carry.inner_carry.z_states.append(new_z)
        
        train_state.system_converged = False
    elif train_state.step % arch_config['convergence_check_interval'] == 0 and len(train_state.gating_grad_variances) > 0:
        avg_grad_variance = np.mean(list(train_state.gating_grad_variances))
        if avg_grad_variance < arch_config['stable_threshold']:
            train_state.system_converged = True
            if rank == 0:
                print(f"Step {train_state.step}: System converged with avg grad variance {avg_grad_variance:.2E}")
    
    # 옵티마이저 스텝
    if world_size > 1:
        for param in train_state.model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad)
    
    lr_this_step = None
    for optim, base_lr in zip(train_state.optimizers, train_state.optimizer_lrs):
        lr_this_step = compute_lr(base_lr, config, train_state)
        for param_group in optim.param_groups:
            param_group['lr'] = lr_this_step
        optim.step()
    
    for optim in train_state.optimizers:
        optim.zero_grad()

    with torch.no_grad():
        train_state.carry.inner_carry.z_states[chosen_module_idx].copy_(final_active_signal)
        other_idx_counter = 0
        for i in range(num_modules):
            if i != chosen_module_idx:
                train_state.carry.inner_carry.z_states[i].copy_(z_others_nograd[other_idx_counter])
                other_idx_counter += 1
    
    if len(metrics) > 0:
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
            # C. WandB 로깅
            reduced_metrics["train/current_module_count"] = num_modules
            reduced_metrics["train/min_predicted_error"] = min_error_this_step
            reduced_metrics["train/hard_problem_threshold"] = train_state.hard_problem_threshold
            reduced_metrics["train/gating_grad_variance"] = current_grad_variance
            return reduced_metrics
def evaluate(config: PretrainConfig, train_state: TrainState, eval_loader: torch.utils.data.DataLoader, eval_metadata: PuzzleDatasetMetadata, rank: int, world_size: int):
    with torch.inference_mode():
        all_preds = {}
        arch_config = config.arch.__pydantic_extra__
        all_metrics = []
        for set_name, batch, global_batch_size in eval_loader:
            batch = {k: v.cuda() for k, v in batch.items()}
            with torch.device("cuda"):
                carry = train_state.model.initial_carry(batch)
            final_active_signal = None
            for _ in range(arch_config['halt_max_steps']):
                carry.inner_carry = train_state.model.model.inner.reset_carry(torch.ones_like(carry.halted), carry.inner_carry) # Always reset
                carry.steps.zero_()
                carry.current_data = batch
                num_modules = len(train_state.model.model.inner.reasoning_modules)
                input_embeddings = train_state.model.model.inner._input_embeddings(carry.current_data["inputs"], carry.current_data.get("puzzle_identifiers"))
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
                carry.inner_carry.z_states[chosen_module_idx].copy_(final_active_signal)
                other_idx_counter = 0
                for i in range(num_modules):
                    if i != chosen_module_idx:
                        carry.inner_carry.z_states[i].copy_(z_others_next[other_idx_counter])
                        other_idx_counter += 1
            if final_active_signal is not None:
                final_logits = train_state.model.model.inner.lm_head(final_active_signal[:, -config.arch.seq_len:, :])
                labels = carry.current_data["labels"]
                mask = labels != IGNORE_LABEL_ID
                loss_counts = mask.sum(-1)
                is_correct = mask & (torch.argmax(final_logits, dim=-1) == labels)
                seq_is_correct = is_correct.sum(-1) == loss_counts
                metrics = {
                    "count": torch.tensor(loss_counts.shape[0], device="cuda", dtype=torch.float32),
                    "exact_accuracy": seq_is_correct.float().sum()
                }
                all_metrics.append(metrics)
                if len(config.eval_save_outputs) > 0:
                    preds = {"logits": final_logits}
                    for collection in (batch, preds):
                        for k, v in collection.items():
                            if k in config.eval_save_outputs:
                                all_preds.setdefault(k, [])
                                all_preds[k].append(v.cpu())
        if len(all_preds) > 0 and config.checkpoint_path is not None:
            all_preds = {k: torch.cat(v, dim=0) for k, v in all_preds.items()}
            os.makedirs(config.checkpoint_path, exist_ok=True)
            torch.save(all_preds, os.path.join(config.checkpoint_path, f"step_{train_state.step}_all_preds.{rank}.pt"))
        if len(all_metrics) > 0:
            metric_keys = sorted(all_metrics[0].keys())
            metric_values = torch.stack([torch.stack([m[k] for k in metric_keys]) for m in all_metrics]).sum(0)
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
        if RANK == 0:
            print(f"Starting Epoch {_iter_id * train_epochs_per_iter}")
        train_state.model.train()
        for _, batch, global_batch_size_effective in train_loader:
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
    if wandb.run and wandb.run.id:
        wandb.finish()
if __name__ == "__main__":
    launch()