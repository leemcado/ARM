from typing import Optional, Any, Sequence, List, Dict
from dataclasses import dataclass, field
import os
import math
import yaml
import shutil
from collections import deque
import multiprocessing as mp
import time

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

# --- 설정 및 상태 클래스 (Pydantic 모델) ---
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
    verbose: int = 0

@dataclass
class TrainState:
    model: nn.Module
    optimizers: Sequence[torch.optim.Optimizer]
    optimizer_lrs: Sequence[float]
    carry: Any
    step: int
    epoch: int
    total_steps: int
    
    gating_grad_variances: deque
    min_predicted_errors: deque
    hard_problem_threshold: float = float('inf')
    system_converged: bool = False
    is_in_stabilization_phase: bool = False
    stabilization_steps_left: int = 0
    module_activation_counts: List[int] = field(default_factory=list)

# --- 유틸리티 함수 ---
def create_dataloader(config: PretrainConfig, split: str, rank: int, world_size: int, **kwargs):
    is_train = split == 'train'
    epochs_per_iter = config.epochs if is_train else 1
    
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
    
    initial_module_count = config.arch.__pydantic_extra__.get('initial_modules', 1)
    
    return TrainState(
        step=0,
        epoch=0,
        total_steps=total_steps,
        model=model,
        optimizers=optimizers,
        optimizer_lrs=optimizer_lrs,
        carry=None,
        gating_grad_variances=deque(maxlen=convergence_interval),
        min_predicted_errors=deque(maxlen=convergence_interval),
        module_activation_counts=[0] * initial_module_count
    )

def train_batch(config: PretrainConfig, train_state: TrainState, batch: Any, global_batch_size: int, rank: int, world_size: int):
    if global_batch_size == 0: return None

    train_state.step += 1
    if train_state.step > train_state.total_steps: return

    arch_config = config.arch.__pydantic_extra__
    device = "cuda"
    batch = {k: v.to(device) for k, v in batch.items()}
    
    if train_state.carry is None:
        train_state.carry = train_state.model.initial_carry(batch)

    current_carry = train_state.carry
    z_f_t = torch.where(current_carry.halted.view(-1, 1, 1), 0.0, current_carry.inner_carry.z_f)
    z_r_states_t = [torch.where(current_carry.halted.view(-1, 1, 1), 0.0, z) for z in current_carry.inner_carry.z_r_states]
    current_carry.steps = torch.where(current_carry.halted, 0, current_carry.steps)
    current_carry.current_data = {k: torch.where(current_carry.halted.view(-1, *([1]*(v.dim()-1))), batch[k], v) for k, v in current_carry.current_data.items()}
    
    input_embeddings = train_state.model.model.inner._input_embeddings(current_carry.current_data["inputs"])
    num_modules = len(train_state.model.model.inner.reasoning_modules)
    predicted_errors = train_state.model.model.inner.thalamus_module(z_f_t, input_embeddings)[:, :num_modules]
    
    with torch.no_grad():
        min_predicted_error_this_step = torch.min(predicted_errors).item()

    active_module_idx = torch.argmin(predicted_errors, dim=1) 
    if train_state.is_in_stabilization_phase and num_modules > 1:
        with torch.no_grad():
            difficulty = torch.min(predicted_errors[:, :-1], dim=1).values
            is_hard_mask = difficulty > train_state.hard_problem_threshold
            new_module_idx = num_modules - 1
            active_module_idx = torch.where(is_hard_mask, new_module_idx, active_module_idx)

    if config.verbose > 0:
        unique_indices, counts = torch.unique(active_module_idx, return_counts=True)
        for idx, count in zip(unique_indices.tolist(), counts.tolist()):
            train_state.module_activation_counts[idx] += count

    cos_sin = train_state.model.model.inner.rotary_emb() if hasattr(train_state.model.model.inner, "rotary_emb") else None
    
    all_z_r_a_t_plus_1 = torch.empty_like(z_r_states_t[0])
    updated_z_r_states = list(z_r_states_t) 

    for i in range(num_modules):
        mask = (active_module_idx == i)
        if not mask.any():
            continue
        
        z_r_input = z_r_states_t[i][mask]
        z_f_input = z_f_t[mask]
        emb_input = input_embeddings[mask]
        
        z_r_a_i_t_plus_1 = train_state.model.model.inner.reasoning_modules[i](
            z_r_input, z_f_input, emb_input, cos_sin=cos_sin
        )
        all_z_r_a_t_plus_1[mask] = z_r_a_i_t_plus_1
        
        temp_z_r = updated_z_r_states[i].clone()
        temp_z_r[mask] = z_r_a_i_t_plus_1.detach()
        updated_z_r_states[i] = temp_z_r

    z_f_t_plus_1 = train_state.model.model.inner.frontal_module(z_f_t, all_z_r_a_t_plus_1, cos_sin=cos_sin)
    
    final_logits = train_state.model.model.inner.lm_head(z_f_t_plus_1)
    q_logits = train_state.model.model.inner.q_head(z_f_t_plus_1[:, 0, :])
    q_halt_logits, q_continue_logits = q_logits[..., 0], q_logits[..., 1]
    
    current_carry.steps += 1
    with torch.no_grad():
        is_last_step = current_carry.steps >= arch_config['halt_max_steps']
        halted = is_last_step | (q_halt_logits > q_continue_logits)
        min_halt_steps = (torch.rand_like(q_halt_logits) < arch_config['halt_exploration_prob']) * torch.randint_like(current_carry.steps, low=2, high=arch_config['halt_max_steps'] + 1)
        current_carry.halted = halted & (current_carry.steps >= min_halt_steps)
    
    freeze_frontal = train_state.is_in_stabilization_phase and (num_modules -1) in active_module_idx
    if freeze_frontal:
        for param in train_state.model.model.inner.frontal_module.parameters(): param.requires_grad_(False)

    lm_q_loss, metrics, lm_loss_per_seq = train_state.model(
        carry=current_carry, final_logits=final_logits, q_halt_logits=q_halt_logits, q_continue_logits=q_continue_logits
    )
    
    predicted_error_for_active = torch.gather(predicted_errors, 1, active_module_idx.unsqueeze(1)).squeeze(1)
    gating_loss = F.mse_loss(predicted_error_for_active, lm_loss_per_seq.detach().to(predicted_error_for_active.dtype))
    
    total_loss = lm_q_loss + gating_loss
    
    total_loss.backward()

    with torch.no_grad():
        gate_grads = [p.grad for p in train_state.model.model.inner.thalamus_module.parameters() if p.grad is not None]
        if gate_grads:
            current_grad_variance = torch.var(torch.cat([g.view(-1) for g in gate_grads])).item()
            train_state.gating_grad_variances.append(current_grad_variance)

    if freeze_frontal:
        for param in train_state.model.model.inner.frontal_module.parameters(): param.requires_grad_(True)
            
    if world_size > 1:
        for param in train_state.model.parameters():
            if param.grad is not None: dist.all_reduce(param.grad)
    
    lr_this_step = None
    for optim, base_lr in zip(train_state.optimizers, train_state.optimizer_lrs):
        lr_this_step = compute_lr(base_lr, config, train_state)
        for param_group in optim.param_groups: param_group['lr'] = lr_this_step
        optim.step()
        optim.zero_grad()

    with torch.no_grad():
        current_carry.inner_carry.z_f = z_f_t_plus_1.detach()
        current_carry.inner_carry.z_r_states = [s.detach() for s in updated_z_r_states]

    train_state.min_predicted_errors.append(min_predicted_error_this_step)
    
    if train_state.is_in_stabilization_phase:
        train_state.stabilization_steps_left -= 1
        if train_state.stabilization_steps_left <= 0:
            train_state.is_in_stabilization_phase = False
            train_state.system_converged = False 
            train_state.gating_grad_variances.clear()
            if rank == 0: print(f"\nStep {train_state.step}: Stabilization phase finished.")
    
    elif num_modules < arch_config['max_modules'] and train_state.step > 0 and train_state.step % arch_config['convergence_check_interval'] == 0:
        if len(train_state.gating_grad_variances) > 0:
            avg_grad_variance = np.mean(list(train_state.gating_grad_variances))
            if avg_grad_variance < arch_config['stable_threshold']:
                if rank == 0:
                    print(f"\nStep {train_state.step}: System converged with avg grad variance {avg_grad_variance:.2E}")
                
                errors_np = np.array(list(train_state.min_predicted_errors))
                train_state.hard_problem_threshold = np.percentile(errors_np, (1 - arch_config['rate_hardprob']) * 100)
                
                if train_state.model.model.inner.add_new_reasoning_module():
                    train_state.is_in_stabilization_phase = True
                    train_state.stabilization_steps_left = arch_config['stabilization_duration']
                    train_state.module_activation_counts.append(0)
                    with torch.no_grad():
                        new_z_r = torch.zeros_like(train_state.carry.inner_carry.z_r_states[0])
                        train_state.carry.inner_carry.z_r_states.append(new_z_r)

    if config.verbose > 0 and train_state.step % config.verbose == 0 and rank == 0:
        total_activations = sum(train_state.module_activation_counts)
        if total_activations > 0:
            print(f"\n--- Module Activation Frequency (Steps {train_state.step - config.verbose + 1}-{train_state.step}) ---")
            for i, count in enumerate(train_state.module_activation_counts):
                percentage = (count / total_activations) * 100
                print(f"    Module {i}: {count} activations ({percentage:.2f}%)")
            print("----------------------------------------------------")
        train_state.module_activation_counts = [0] * len(train_state.module_activation_counts)


    if rank == 0:
        count = metrics.get("count", 1.0).item()
        if count == 0: count = 1
        
        reduced_metrics = {
            "train/accuracy": metrics.get("exact_accuracy", torch.tensor(0.0)).item() / count,
            "train/similarity": metrics.get("similarity", torch.tensor(0.0)).item() / count,
            "train/lm_loss": metrics.get('lm_loss', torch.tensor(0.0)).item() / global_batch_size,
            "train/gating_loss": gating_loss.item() / global_batch_size,
            "train/q_halt_loss": metrics.get('q_halt_loss', torch.tensor(0.0)).item() / global_batch_size,
            "train/q_continue_loss": metrics.get('q_continue_loss', torch.tensor(0.0)).item() / global_batch_size,
            "train/lr": lr_this_step,
            "train/current_module_count": num_modules,
            "train/hard_problem_threshold": train_state.hard_problem_threshold,
            "train/min_predicted_error": min_predicted_error_this_step,
        }
        if train_state.gating_grad_variances:
            reduced_metrics["train/gating_grad_variance"] = train_state.gating_grad_variances[-1]
        return reduced_metrics

def evaluate(config: PretrainConfig, model: nn.Module, eval_loader: torch.utils.data.DataLoader, eval_metadata: PuzzleDatasetMetadata, rank: int, world_size: int):
    model.eval()
    all_metrics_list = []
    arch_config = config.arch.__pydantic_extra__
    
    with torch.inference_mode():
        for set_name, batch, global_batch_size in eval_loader:
            if global_batch_size == 0: continue
            batch = {k: v.cuda() for k, v in batch.items()}
            carry = model.initial_carry(batch)
            
            z_f_t_plus_1 = carry.inner_carry.z_f

            for t in range(arch_config['halt_max_steps']):
                z_f_t = carry.inner_carry.z_f
                z_r_states_t = carry.inner_carry.z_r_states
                
                input_embeddings = model.model.inner._input_embeddings(carry.current_data["inputs"])
                num_modules = len(model.model.inner.reasoning_modules)
                
                predicted_errors = model.model.inner.thalamus_module(z_f_t, input_embeddings)[:, :num_modules]
                active_module_idx = torch.argmin(predicted_errors, dim=1)

                cos_sin = model.model.inner.rotary_emb() if hasattr(model.model.inner, "rotary_emb") else None
                
                all_z_r_a_t_plus_1 = torch.empty_like(z_r_states_t[0])
                for i in range(num_modules):
                    mask = (active_module_idx == i)
                    if not mask.any(): continue
                    all_z_r_a_t_plus_1[mask] = model.model.inner.reasoning_modules[i](z_r_states_t[i][mask], z_f_t[mask], input_embeddings[mask], cos_sin=cos_sin)
                
                z_f_t_plus_1 = model.model.inner.frontal_module(z_f_t, all_z_r_a_t_plus_1, cos_sin=cos_sin)
                
                carry.inner_carry.z_f = z_f_t_plus_1
                for i in range(num_modules):
                    mask = (active_module_idx == i)
                    if not mask.any(): continue
                    temp_z_r = carry.inner_carry.z_r_states[i].clone()
                    temp_z_r[mask] = all_z_r_a_t_plus_1[mask]
                    carry.inner_carry.z_r_states[i] = temp_z_r

            final_logits = model.model.inner.lm_head(z_f_t_plus_1)
            labels = carry.current_data["labels"]
            mask = labels != IGNORE_LABEL_ID
            loss_counts = mask.sum(-1)
            
            preds = torch.argmax(final_logits, dim=-1)
            is_correct = mask & (preds == labels)
            seq_is_correct = is_correct.sum(-1) == loss_counts
            
            pixel_similarity = is_correct.sum(-1).float() / loss_counts.float().clamp_min(1)

            metrics = {
                "count": torch.tensor(loss_counts.shape[0], device="cuda", dtype=torch.float32),
                "exact_accuracy": seq_is_correct.float().sum(),
                "similarity": pixel_similarity.sum(),
            }
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
        if count == 0: count = 1
        final_metrics = {f"eval/{k}": v / count for k, v in total_metrics.items()}
        return final_metrics
    return None

def evaluate_worker(rank: int, world_size: int, config: PretrainConfig, model_state_dict: Dict[str, torch.Tensor], step: int, epoch: int, metrics_queue: mp.Queue):
    if "LOCAL_RANK" not in os.environ:
        torch.cuda.set_device(rank)
    
    _, eval_metadata = create_dataloader(config, "test", rank=rank, world_size=world_size)
    eval_loader, _ = create_dataloader(config, "test", rank=rank, world_size=world_size) 
    
    model, _, _ = create_model(config, eval_metadata, world_size)
    model.load_state_dict(model_state_dict)

    eval_metrics = evaluate(config, model, eval_loader, eval_metadata, rank=rank, world_size=world_size)
    
    if rank == 0 and eval_metrics is not None:
        # wandb 로깅 대신 Queue에 결과를 넣음
        metrics_queue.put((step, epoch, eval_metrics))


def rank_zero_only():
    return not dist.is_initialized() or dist.get_rank() == 0

def save_train_state(config: PretrainConfig, train_state: TrainState):
    if config.checkpoint_path is None or not rank_zero_only(): return
    os.makedirs(config.checkpoint_path, exist_ok=True)
    model = train_state.model
    if hasattr(model, "_orig_mod"): model = model._orig_mod
    
    model_state = model.state_dict()
    if hasattr(model, 'module'):
        model_state = model.module.state_dict()
    torch.save(model_state, os.path.join(config.checkpoint_path, f"epoch_{train_state.epoch}_step_{train_state.step}.pth"))

def save_code_and_config(config: PretrainConfig, hydra_config: DictConfig):
    if config.checkpoint_path is None or not rank_zero_only() or wandb.run is None: return
    os.makedirs(config.checkpoint_path, exist_ok=True)
    with open(os.path.join(config.checkpoint_path, "all_config.yaml"), "w") as f:
        yaml.dump(OmegaConf.to_container(hydra_config, resolve=True), f)

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

@hydra.main(config_path="config", config_name="cfg_pretrain", version_base=None)
def launch(hydra_config: DictConfig):
    mp.set_start_method("spawn", force=True)
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
    
    train_state = init_train_state(config, train_metadata, world_size=WORLD_SIZE)

    progress_bar = None
    eval_process = None
    metrics_queue = None

    steps_per_epoch = int(train_metadata.total_groups * train_metadata.mean_puzzle_examples / config.global_batch_size)

    if rank_zero_only():
        progress_bar = tqdm.tqdm(total=train_state.total_steps, desc="Training")
        wandb.init(project=config.project_name, name=config.run_name, config=config.model_dump())
        wandb.watch(train_state.model, log_freq=100)
        wandb.log({"num_params": sum(p.numel() for p in train_state.model.parameters())}, step=0)
        save_code_and_config(config, hydra_config)
        metrics_queue = mp.Queue()

    train_state.model.train()
    for _, batch, global_batch_size_effective in train_loader:
        if train_state.step >= train_state.total_steps: break
        
        # 메인 프로세스에서 Queue를 확인하고 결과가 있으면 로깅
        if RANK == 0 and not metrics_queue.empty():
            step, epoch, eval_metrics = metrics_queue.get()
            eval_metrics_with_epoch = {**eval_metrics, "epoch": epoch}
            wandb.log(eval_metrics_with_epoch, step=step)
            print(f"Evaluation results from epoch {epoch} logged.")

        metrics = train_batch(config, train_state, batch, global_batch_size_effective, rank=RANK, world_size=WORLD_SIZE)
        
        current_epoch = (train_state.step -1) // steps_per_epoch + 1
        train_state.epoch = current_epoch
        
        if rank_zero_only() and metrics is not None:
            wandb.log({**metrics, "epoch": current_epoch}, step=train_state.step)
            progress_bar.update(1)
    
        if config.eval_interval and train_state.step > 0 and train_state.step % (steps_per_epoch * config.eval_interval) == 0:
            if eval_process is not None and eval_process.is_alive():
                print(f"\nWaiting for the previous evaluation process to finish...")
                eval_process.join()

            model_state_cpu = {k: v.cpu() for k, v in train_state.model.state_dict().items()}

            eval_process = mp.Process(target=evaluate_worker, args=(
                RANK, WORLD_SIZE, config, model_state_cpu, train_state.step, current_epoch, metrics_queue
            ))
            eval_process.start()
            
            if config.checkpoint_every_eval:
                save_train_state(config, train_state)
    
    if rank_zero_only():
        progress_bar.close()
        # 마지막으로 큐에 남아있는 결과 처리
        while not metrics_queue.empty():
            step, epoch, eval_metrics = metrics_queue.get()
            eval_metrics_with_epoch = {**eval_metrics, "epoch": epoch}
            wandb.log(eval_metrics_with_epoch, step=step)
            print(f"Final evaluation results from epoch {epoch} logged.")
        save_train_state(config, train_state)
    
    if eval_process is not None:
        eval_process.join()
    
    if dist.is_initialized(): dist.destroy_process_group()
    if wandb.run: wandb.finish()

if __name__ == "__main__":
    launch()