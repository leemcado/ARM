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
import numpy as np
import tqdm
import wandb
import coolname
import hydra
import pydantic
from omegaconf import DictConfig, OmegaConf

from adam_atan2 import AdamATan2
from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig, PuzzleDatasetMetadata
from utils.functions import load_model_class, get_model_source_path
from models.losses import IGNORE_LABEL_ID, ARMLossHead
from models.arm.arm_v1 import ARM, ARMCarry, ARMInnerCarry

# --- 설정 및 상태 클래스 (변경 없음) ---
class LossConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')
    name: str

class ArchConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')
    name: str
    loss: LossConfig

class PretrainConfig(pydantic.BaseModel):
    arch: ArchConfig
    data_path: str
    global_batch_size: int
    epochs: int
    lr: float
    lr_min_ratio: float
    lr_warmup_steps: int
    weight_decay: float
    beta1: float
    beta2: float
    project_name: Optional[str] = None
    run_name: Optional[str] = None
    checkpoint_path: Optional[str] = None
    seed: int = 0
    checkpoint_every_eval: bool = False
    eval_interval: Optional[int] = None
    eval_save_outputs: List[str] = []

@dataclass
class TrainState:
    model: nn.Module
    optimizers: Sequence[torch.optim.Optimizer]
    optimizer_lrs: Sequence[float]
    carry: Any  # carry 객체를 TrainState가 직접 관리
    step: int
    total_steps: int
    
    gating_grad_variances: deque
    min_predicted_errors: deque
    hard_problem_threshold: float = float('inf')
    system_converged: bool = False
    is_in_stabilization_phase: bool = False
    stabilization_steps_left: int = 0

# --- 유틸리티 함수 (변경 없음) ---
def create_dataloader(config: PretrainConfig, split: str, rank: int, world_size: int, **kwargs):
    is_train = split == 'train'
    # epochs_per_iter: eval_interval이 None일 경우 전체 epochs를 사용하도록 수정
    epochs_per_iter = config.eval_interval if is_train and config.eval_interval is not None else (config.epochs if is_train else 1)
    
    dataset_config = PuzzleDatasetConfig(
        seed=config.seed,
        dataset_path=config.data_path,
        global_batch_size=config.global_batch_size,
        test_set_mode=not is_train,
        epochs_per_iter=epochs_per_iter,
        rank=rank,
        num_replicas=world_size
    )
    dataset = PuzzleDataset(dataset_config, split=split)
    dataloader = DataLoader(
        dataset, batch_size=None, num_workers=1, prefetch_factor=8,
        pin_memory=True, persistent_workers=True if is_train else False
    )
    return dataloader, dataset.metadata

def cosine_schedule_with_warmup_lr_lambda(current_step: int, *, base_lr: float, num_warmup_steps: int, num_training_steps: int, min_ratio: float = 0.0, num_cycles: float = 0.5):
    if current_step < num_warmup_steps:
        return base_lr * float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return base_lr * (min_ratio + max(0.0, (1 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))))

def compute_lr(base_lr: float, config: PretrainConfig, train_state: TrainState):
    return cosine_schedule_with_warmup_lr_lambda(
        current_step=train_state.step,
        base_lr=base_lr,
        num_warmup_steps=round(config.lr_warmup_steps),
        num_training_steps=train_state.total_steps,
        min_ratio=config.lr_min_ratio
    )

# --- 모델 및 TrainState 초기화 함수 (carry=None 추가) ---
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
    optimizers = [AdamATan2(model.parameters(), lr=0, weight_decay=config.weight_decay, betas=(config.beta1, config.beta2))]
    optimizer_lrs = [config.lr]
    return model, optimizers, optimizer_lrs

def init_train_state(config: PretrainConfig, train_metadata: PuzzleDatasetMetadata, world_size: int):
    total_steps = int(config.epochs * train_metadata.total_groups * train_metadata.mean_puzzle_examples / config.global_batch_size)
    model, optimizers, optimizer_lrs = create_model(config, train_metadata, world_size=world_size)
    convergence_interval = config.arch.__pydantic_extra__.get('convergence_check_interval', 1000)
    return TrainState(
        step=0, total_steps=total_steps, model=model, optimizers=optimizers, optimizer_lrs=optimizer_lrs,
        carry=None, # carry는 첫 배치에서 초기화
        gating_grad_variances=deque(maxlen=convergence_interval),
        min_predicted_errors=deque(maxlen=convergence_interval)
    )

# --- train_batch 함수 최종 수정 ---
# 각 호출이 하나의 시간 스텝(t) 역할을 하도록 수정
def train_batch(config: PretrainConfig, train_state: TrainState, batch: Any, global_batch_size: int, rank: int, world_size: int):
    train_state.step += 1
    if train_state.step > train_state.total_steps: return

    arch_config = config.arch.__pydantic_extra__
    device = "cuda"
    batch = {k: v.to(device) for k, v in batch.items()}
    
    # Carry가 없으면(첫 스텝) 초기화
    if train_state.carry is None:
        train_state.carry = train_state.model.initial_carry(batch)

    # --- ACT 로직: 멈춘 시퀀스 리셋 ---
    # 멈춘(halted) 시퀀스의 데이터를 새 데이터로 교체하고, 히든 스테이트와 스텝 카운터를 0으로 리셋
    current_carry = train_state.carry
    z_f_t = torch.where(current_carry.halted.view(-1, 1, 1), 0.0, current_carry.inner_carry.z_f)
    z_r_states_t = [torch.where(current_carry.halted.view(-1, 1, 1), 0.0, z) for z in current_carry.inner_carry.z_r_states]
    current_carry.steps = torch.where(current_carry.halted, 0, current_carry.steps)
    current_carry.current_data = {k: torch.where(current_carry.halted.view(-1, *([1]*(v.dim()-1))), batch[k], v) for k, v in current_carry.current_data.items()}
    
    # --- 순전파 (1 time-step) ---
    input_embeddings = train_state.model.model.inner._input_embeddings(current_carry.current_data["inputs"])
    num_modules = len(train_state.model.model.inner.reasoning_modules)
    predicted_errors = train_state.model.model.inner.thalamus_module(z_f_t, input_embeddings)[:, :num_modules]
    
    with torch.no_grad():
        active_module_idx = torch.argmin(predicted_errors, dim=1)[0].item()
        if train_state.is_in_stabilization_phase and num_modules > 1:
            difficulty = torch.min(predicted_errors[:, :-1], dim=1).values
            if difficulty.max().item() > train_state.hard_problem_threshold:
                active_module_idx = num_modules - 1

    cos_sin = train_state.model.model.inner.rotary_emb() if hasattr(train_state.model.model.inner, "rotary_emb") else None
    
    z_r_a_t_plus_1 = train_state.model.model.inner.reasoning_modules[active_module_idx](z_r_states_t[active_module_idx], z_f_t, input_embeddings, cos_sin=cos_sin)
    z_f_t_plus_1 = train_state.model.model.inner.frontal_module(z_f_t, z_r_a_t_plus_1, cos_sin=cos_sin)
    
    final_logits = train_state.model.model.inner.lm_head(z_f_t_plus_1)
    q_logits = train_state.model.model.inner.q_head(z_f_t_plus_1[:, 0, :])
    q_halt_logits, q_continue_logits = q_logits[..., 0], q_logits[..., 1]
    
    current_carry.steps += 1
    with torch.no_grad():
        is_last_step = current_carry.steps >= arch_config['halt_max_steps']
        halted = is_last_step | (q_halt_logits > q_continue_logits)
        min_halt_steps = (torch.rand_like(q_halt_logits) < arch_config['halt_exploration_prob']) * torch.randint_like(current_carry.steps, low=2, high=arch_config['halt_max_steps'] + 1)
        current_carry.halted = halted & (current_carry.steps >= min_halt_steps)
    
    # --- 역전파 ---
    freeze_frontal = train_state.is_in_stabilization_phase and (active_module_idx == num_modules - 1)
    if freeze_frontal:
        for param in train_state.model.model.inner.frontal_module.parameters(): param.requires_grad_(False)

    lm_q_loss, metrics = train_state.model(carry=current_carry, final_logits=final_logits, q_halt_logits=q_halt_logits, q_continue_logits=q_continue_logits)
    predicted_error_for_active = predicted_errors[:, active_module_idx]
    actual_lm_loss = metrics.get('lm_loss', torch.tensor(0.0, device=device))
    gating_loss = F.mse_loss(predicted_error_for_active, actual_lm_loss.detach())
    total_loss = lm_q_loss + gating_loss
    
    total_loss.backward()

    # 수렴 판단을 위한 그래디언트 분산 계산 (역전파 직후)
    with torch.no_grad():
        gate_grads = [p.grad for p in train_state.model.model.inner.thalamus_module.parameters() if p.grad is not None]
        if gate_grads:
            current_grad_variance = torch.var(torch.cat([g.view(-1) for g in gate_grads])).item()
            train_state.gating_grad_variances.append(current_grad_variance)

    if freeze_frontal:
        for param in train_state.model.model.inner.frontal_module.parameters(): param.requires_grad_(True)
            
    # --- 옵티마이저 스텝 ---
    if world_size > 1:
        for param in train_state.model.parameters():
            if param.grad is not None: dist.all_reduce(param.grad)
    
    lr_this_step = None
    for optim, base_lr in zip(train_state.optimizers, train_state.optimizer_lrs):
        lr_this_step = compute_lr(base_lr, config, train_state)
        for param_group in optim.param_groups: param_group['lr'] = lr_this_step
        optim.step()
        optim.zero_grad()

    # --- 다음 스텝을 위한 Carry 상태 업데이트 (그래프 분리) ---
    with torch.no_grad():
        current_carry.inner_carry.z_f = z_f_t_plus_1.detach()
        z_r_states_t[active_module_idx] = z_r_a_t_plus_1.detach()
        # 비활성 모듈은 이전 상태를 그대로 유지(detach)
        for i in range(len(z_r_states_t)):
            if i != active_module_idx:
                z_r_states_t[i] = z_r_states_t[i].detach()
        current_carry.inner_carry.z_r_states = z_r_states_t

    # --- 성장 메커니즘 상태 머신 ---
    train_state.min_predicted_errors.append(torch.min(predicted_errors).item())
    
    if train_state.is_in_stabilization_phase:
        train_state.stabilization_steps_left -= 1
        if train_state.stabilization_steps_left <= 0:
            train_state.is_in_stabilization_phase = False
            train_state.system_converged = False
            if rank == 0: print(f"\nStep {train_state.step}: Stabilization phase finished.")
    elif train_state.system_converged:
        errors_np = np.array(list(train_state.min_predicted_errors))
        train_state.hard_problem_threshold = np.percentile(errors_np, (1 - arch_config['rate_hardprob']) * 100)
        
        if train_state.model.model.inner.add_new_reasoning_module():
            train_state.is_in_stabilization_phase = True
            train_state.stabilization_steps_left = arch_config['stabilization_duration']
            # 새 모듈에 대한 히든 스테이트를 carry에 추가
            with torch.no_grad():
                new_z_r = torch.zeros_like(train_state.carry.inner_carry.z_r_states[0])
                train_state.carry.inner_carry.z_r_states.append(new_z_r)
        
        train_state.system_converged = False
    elif train_state.step > 0 and train_state.step % arch_config['convergence_check_interval'] == 0 and len(train_state.gating_grad_variances) > 0:
        avg_grad_variance = np.mean(list(train_state.gating_grad_variances))
        if avg_grad_variance < arch_config['stable_threshold']:
            train_state.system_converged = True
            if rank == 0:
                print(f"\nStep {train_state.step}: System converged with avg grad variance {avg_grad_variance:.2E}")

    # --- 메트릭 로깅 ---
    if rank == 0:
        count = metrics.get("count", 1.0).item()
        reduced_metrics = {f"train/{k}": v.item() / count for k, v in metrics.items() if k != "count"}
        reduced_metrics["train/lm_loss"] = metrics.get('lm_loss', torch.tensor(0.0)).item() / global_batch_size
        reduced_metrics["train/gating_loss"] = gating_loss.item() / global_batch_size
        reduced_metrics["train/lr"] = lr_this_step
        reduced_metrics["train/current_module_count"] = num_modules
        reduced_metrics["train/hard_problem_threshold"] = train_state.hard_problem_threshold
        if train_state.gating_grad_variances:
            reduced_metrics["train/gating_grad_variance"] = train_state.gating_grad_variances[-1]
        return reduced_metrics

# --- 평가 함수 최종 수정 ---
def evaluate(config: PretrainConfig, train_state: TrainState, eval_loader: torch.utils.data.DataLoader, eval_metadata: PuzzleDatasetMetadata, rank: int, world_size: int):
    train_state.model.eval()
    all_metrics_list = []
    arch_config = config.arch.__pydantic_extra__
    
    with torch.inference_mode():
        for set_name, batch, global_batch_size in eval_loader:
            batch = {k: v.cuda() for k, v in batch.items()}
            carry = train_state.model.initial_carry(batch)

            for t in range(arch_config['halt_max_steps']):
                z_f_t = carry.inner_carry.z_f
                z_r_states_t = carry.inner_carry.z_r_states
                
                input_embeddings = train_state.model.model.inner._input_embeddings(carry.current_data["inputs"])
                num_modules = len(train_state.model.model.inner.reasoning_modules)
                
                predicted_errors = train_state.model.model.inner.thalamus_module(z_f_t, input_embeddings)[:, :num_modules]
                active_module_idx = torch.argmin(predicted_errors, dim=1)[0].item()

                cos_sin = train_state.model.model.inner.rotary_emb() if hasattr(train_state.model.model.inner, "rotary_emb") else None

                z_r_a_t_plus_1 = train_state.model.model.inner.reasoning_modules[active_module_idx](z_r_states_t[active_module_idx], z_f_t, input_embeddings, cos_sin=cos_sin)
                z_f_t_plus_1 = train_state.model.model.inner.frontal_module(z_f_t, z_r_a_t_plus_1, cos_sin=cos_sin)
                
                carry.inner_carry.z_f = z_f_t_plus_1
                z_r_states_t[active_module_idx] = z_r_a_t_plus_1
                carry.inner_carry.z_r_states = z_r_states_t
                
                q_logits = train_state.model.model.inner.q_head(z_f_t_plus_1[:, 0, :])
                # 평가 시에는 argmax로 결정 (탐험 없음)
                if q_logits[..., 0] > q_logits[..., 1] or (t == arch_config['halt_max_steps'] - 1):
                    break
            
            final_logits = train_state.model.model.inner.lm_head(z_f_t_plus_1)
            labels = carry.current_data["labels"]
            mask = labels != IGNORE_LABEL_ID
            loss_counts = mask.sum(-1)
            is_correct = mask & (torch.argmax(final_logits, dim=-1) == labels)
            seq_is_correct = is_correct.sum(-1) == loss_counts
            
            metrics = {"count": torch.tensor(loss_counts.shape[0], device="cuda", dtype=torch.float32), "exact_accuracy": seq_is_correct.float().sum()}
            all_metrics_list.append(metrics)
            
    if not all_metrics_list: return None

    total_metrics = {k: 0.0 for k in all_metrics_list[0].keys()}
    for metrics in all_metrics_list:
        for k, v in metrics.items(): total_metrics[k] += v
    
    if world_size > 1:
        metric_tensor = torch.tensor([total_metrics[k] for k in sorted(total_metrics.keys())], device="cuda")
        dist.reduce(metric_tensor, dst=0)
        if rank == 0:
            reduced_vals = metric_tensor.cpu().numpy()
            for i, k in enumerate(sorted(total_metrics.keys())): total_metrics[k] = reduced_vals[i]

    if rank == 0:
        count = total_metrics.pop("count")
        final_metrics = {f"eval/{k}": v / count for k, v in total_metrics.items()}
        return final_metrics
    return None

# --- 파일 저장 및 W&B 로깅, 메인 실행 함수 (변경 없음) ---
def save_train_state(config: PretrainConfig, train_state: TrainState):
    if config.checkpoint_path is None or rank_zero_only(): return
    os.makedirs(config.checkpoint_path, exist_ok=True)
    model_state = train_state.model.state_dict()
    if hasattr(train_state.model, 'module'): model_state = train_state.model.module.state_dict()
    torch.save(model_state, os.path.join(config.checkpoint_path, f"step_{train_state.step}.pth"))

def save_code_and_config(config: PretrainConfig):
    if config.checkpoint_path is None or not rank_zero_only() or wandb.run is None: return
    os.makedirs(config.checkpoint_path, exist_ok=True)
    code_files = [get_model_source_path(config.arch.name), get_model_source_path(config.arch.loss.name)]
    for code_file in code_files:
        if code_file and os.path.exists(code_file):
            shutil.copy(code_file, os.path.join(config.checkpoint_path, os.path.basename(code_file)))
    with open(os.path.join(config.checkpoint_path, "all_config.yaml"), "w") as f:
        yaml.dump(OmegaConf.to_container(config, resolve=True), f)

def load_synced_config(hydra_config: DictConfig, rank: int, world_size: int) -> PretrainConfig:
    objects = [None]
    if rank == 0:
        config = PretrainConfig(**OmegaConf.to_container(hydra_config, resolve=True))
        if config.project_name is None: config.project_name = f"{os.path.basename(config.data_path).capitalize()}-ARM"
        if config.run_name is None: config.run_name = f"{config.arch.name.split('@')[-1]}-{coolname.generate_slug(2)}"
        if config.checkpoint_path is None: config.checkpoint_path = os.path.join("checkpoints", config.project_name, config.run_name)
        objects = [config]
    if world_size > 1: dist.broadcast_object_list(objects, src=0)
    return objects[0]

def rank_zero_only():
    return not dist.is_initialized() or dist.get_rank() == 0

@hydra.main(config_path="config", config_name="cfg_pretrain", version_base=None)
def launch(hydra_config: DictConfig):
    RANK, WORLD_SIZE = 0, 1
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        RANK = dist.get_rank()
        WORLD_SIZE = dist.get_world_size()
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    config = load_synced_config(hydra_config, rank=RANK, world_size=WORLD_SIZE)
    torch.manual_seed(config.seed + RANK)
    np.random.seed(config.seed + RANK)

    train_loader, train_metadata = create_dataloader(config, "train", rank=RANK, world_size=WORLD_SIZE)
    eval_loader, eval_metadata = create_dataloader(config, "test", rank=RANK, world_size=WORLD_SIZE)
    
    train_state = init_train_state(config, train_metadata, world_size=WORLD_SIZE)

    progress_bar = None
    if rank_zero_only():
        progress_bar = tqdm.tqdm(total=train_state.total_steps, desc="Training")
        wandb.init(project=config.project_name, name=config.run_name, config=OmegaConf.to_container(config, resolve=True))
        wandb.watch(train_state.model, log_freq=100)
        wandb.log({"num_params": sum(p.numel() for p in train_state.model.parameters())}, step=0)
        save_code_and_config(config)

    for epoch in range(config.epochs):
        if rank_zero_only(): print(f"\nStarting Epoch {epoch+1}/{config.epochs}")
        
        train_state.model.train()
        for _, batch, global_batch_size_effective in train_loader:
            if train_state.step >= train_state.total_steps: break
            
            metrics = train_batch(config, train_state, batch, global_batch_size_effective, rank=RANK, world_size=WORLD_SIZE)
            
            if rank_zero_only() and metrics is not None:
                wandb.log(metrics, step=train_state.step)
                progress_bar.update(1)
        
        if train_state.step >= train_state.total_steps: break

        if config.eval_interval and (epoch + 1) % config.eval_interval == 0:
            eval_metrics = evaluate(config, train_state, eval_loader, eval_metadata, rank=RANK, world_size=WORLD_SIZE)
            if rank_zero_only() and eval_metrics is not None:
                wandb.log(eval_metrics, step=train_state.step)
            
            if config.checkpoint_every_eval:
                save_train_state(config, train_state)

    if rank_zero_only():
        progress_bar.close()
        save_train_state(config, train_state)
    
    if dist.is_initialized(): dist.destroy_process_group()
    if wandb.run: wandb.finish()

if __name__ == "__main__":
    launch()