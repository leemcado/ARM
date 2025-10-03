from typing import Any, Tuple, Dict, Sequence, Optional, List

import torch
import torch.nn.functional as F
from torch import nn


IGNORE_LABEL_ID = -100


def s(x, epsilon=1e-30):
    return torch.where(
        x<0,
        1/(1-x+ epsilon),
        x + 1
    )


def log_stablemax(x, dim=-1):
    s_x = s(x)
    return torch.log(s_x/torch.sum(s_x, dim=dim, keepdim=True))


def stablemax_cross_entropy(logits, labels, ignore_index: int = -100):
    logprobs = log_stablemax(logits.to(torch.float64), dim=-1)

    valid_mask = labels != ignore_index
    transformed_labels = torch.where(valid_mask, labels, 0)
    prediction_logprobs = torch.gather(logprobs, index=transformed_labels.to(torch.long).unsqueeze(-1), dim=-1).squeeze(-1)

    return -torch.where(valid_mask, prediction_logprobs, 0)


def softmax_cross_entropy(logits, labels, ignore_index: int = -100):
    # Cast logits to f32
    # Flatten logits
    return F.cross_entropy(logits.to(torch.float32).view(-1, logits.shape[-1]), labels.to(torch.long).view(-1), ignore_index=ignore_index, reduction="none").view(labels.shape)


# 기존 ACTLossHead는 HRM을 위해 그대로 둡니다.
class ACTLossHead(nn.Module):
    def __init__(self, model: nn.Module, loss_type: str):
        super().__init__()
        self.model = model
        self.loss_fn = globals()[loss_type]
        
    def initial_carry(self, *args, **kwargs):
        return self.model.initial_carry(*args, **kwargs)

    def forward(
        self,
        return_keys: Sequence[str],
        # Model args
        **model_kwargs,
    ) -> Tuple[Any, torch.Tensor, Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]], torch.Tensor]:
        new_carry, outputs = self.model(**model_kwargs)
        labels = new_carry.current_data["labels"]

        with torch.no_grad():
            mask = labels != IGNORE_LABEL_ID
            loss_counts = mask.sum(-1)
            loss_divisor = loss_counts.clamp_min(1).unsqueeze(-1)

            is_correct = mask & (torch.argmax(outputs["logits"], dim=-1) == labels)
            seq_is_correct = is_correct.sum(-1) == loss_counts
            
            valid_metrics = new_carry.halted & (loss_counts > 0)
            metrics = {
                "count": valid_metrics.sum(),
                "accuracy":       torch.where(valid_metrics, (is_correct.to(torch.float32) / loss_divisor).sum(-1), 0).sum(),
                "exact_accuracy": (valid_metrics & seq_is_correct).sum(),
                "q_halt_accuracy": (valid_metrics & ((outputs["q_halt_logits"] >= 0) == seq_is_correct)).sum(),
                "steps":          torch.where(valid_metrics, new_carry.steps, 0).sum(),
            }

        lm_loss = (self.loss_fn(outputs["logits"], labels, ignore_index=IGNORE_LABEL_ID) / loss_divisor).sum()
        q_halt_loss = F.binary_cross_entropy_with_logits(outputs["q_halt_logits"], seq_is_correct.to(outputs["q_halt_logits"].dtype), reduction="sum")

        metrics.update({
            "lm_loss": lm_loss.detach(),
            "q_halt_loss": q_halt_loss.detach(),
        })

        q_continue_loss = 0
        if "target_q_continue" in outputs:
            q_continue_loss = F.binary_cross_entropy_with_logits(outputs["q_continue_logits"], outputs["target_q_continue"], reduction="sum")
            metrics["q_continue_loss"] = q_continue_loss.detach()

        detached_outputs = {k: outputs[k].detach() for k in return_keys if k in outputs}

        return new_carry, lm_loss + 0.5 * (q_halt_loss + q_continue_loss), metrics, detached_outputs, new_carry.halted.all()


# +++ ARM을 위한 새로운 Loss Head 클래스 수정 +++
class ARMLossHead(nn.Module):
    def __init__(self, model: nn.Module, loss_type: str):
        super().__init__()
        self.model = model
        self.loss_fn = globals()[loss_type]
    
    def initial_carry(self, *args, **kwargs):
        return self.model.initial_carry(*args, **kwargs)

    def forward(
        self,
        # pretrain.py의 커스텀 훈련 루프에 의해 직접 제어되도록 인터페이스를 변경합니다.
        carry: Any,
        # 외부에서 계산된 값들을 직접 인자로 받습니다.
        final_logits: torch.Tensor, # 전두엽 모듈의 최종 출력 로짓
        q_halt_logits: torch.Tensor,
        q_continue_logits: torch.Tensor,
        target_q_continue: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]: # 반환 값에서 carry, detached_outputs 등을 제거하여 단순화
        
        labels = carry.current_data["labels"]

        # --- 1. LM Loss 계산 ---
        # pretrain.py에서 활성 모듈의 로짓 하나만 넘어오므로, 루프 없이 직접 계산합니다.
        mask = labels != IGNORE_LABEL_ID
        loss_counts = mask.sum(-1)
        loss_divisor = loss_counts.clamp_min(1) # B, 
        
        lm_loss_per_seq = self.loss_fn(final_logits, labels, ignore_index=IGNORE_LABEL_ID).sum(-1) / loss_divisor
        total_lm_loss = torch.where(carry.halted, lm_loss_per_seq, 0).sum()

        # --- 2. 정확도 및 Q-러닝 관련 메트릭 계산 ---
        with torch.no_grad():
            is_correct = mask & (torch.argmax(final_logits, dim=-1) == labels)
            seq_is_correct = is_correct.sum(-1) == loss_counts
            
            valid_metrics = carry.halted & (loss_counts > 0)
            metrics = {
                "count": valid_metrics.sum(),
                "exact_accuracy": (valid_metrics & seq_is_correct).sum(),
                "steps": torch.where(valid_metrics, carry.steps, 0).sum(),
            }

        # --- 3. Q-러닝 손실 계산 ---
        q_halt_loss = F.binary_cross_entropy_with_logits(q_halt_logits, seq_is_correct.to(q_halt_logits.dtype), reduction="sum")

        metrics.update({
            "lm_loss": total_lm_loss.detach(), # 시상 모듈 학습을 위해 detach된 lm_loss를 metrics에 포함
            "q_halt_loss": q_halt_loss.detach(),
        })

        q_continue_loss = 0
        if target_q_continue is not None:
            q_continue_loss = F.binary_cross_entropy_with_logits(q_continue_logits, target_q_continue, reduction="sum")
            metrics["q_continue_loss"] = q_continue_loss.detach()

        # pretrain.py에서 gating_loss가 별도로 추가될 것이므로, 여기서는 LM Loss와 Q-Loss만 합산
        final_loss = total_lm_loss + 0.5 * (q_halt_loss + q_continue_loss)

        return final_loss, metrics