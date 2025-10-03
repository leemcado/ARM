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
    return F.cross_entropy(logits.to(torch.float32).view(-1, logits.shape[-1]), labels.to(torch.long).view(-1), ignore_index=ignore_index, reduction="none").view(labels.shape)


class ACTLossHead(nn.Module):
    # (이전 버전의 HRM을 위한 클래스이므로 변경 없음)
    def __init__(self, model: nn.Module, loss_type: str):
        super().__init__()
        self.model = model
        self.loss_fn = globals()[loss_type]
        
    def initial_carry(self, *args, **kwargs):
        return self.model.initial_carry(*args, **kwargs)

    def forward(
        self,
        return_keys: Sequence[str],
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


class ARMLossHead(nn.Module):
    def __init__(self, model: nn.Module, loss_type: str):
        super().__init__()
        self.model = model
        self.loss_fn = globals()[loss_type]
    
    def initial_carry(self, *args, **kwargs):
        return self.model.initial_carry(*args, **kwargs)

    def forward(
        self,
        carry: Any,
        final_logits: torch.Tensor,
        q_halt_logits: torch.Tensor,
        q_continue_logits: torch.Tensor,
        target_q_continue: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        
        labels = carry.current_data["labels"]

        mask = labels != IGNORE_LABEL_ID
        loss_counts = mask.sum(-1)
        loss_divisor = loss_counts.clamp_min(1)
        
        lm_loss_per_seq = self.loss_fn(final_logits, labels, ignore_index=IGNORE_LABEL_ID).sum(-1) / loss_divisor
        total_lm_loss = torch.where(carry.halted, lm_loss_per_seq, 0).sum()

        with torch.no_grad():
            preds = torch.argmax(final_logits, dim=-1)
            is_correct = mask & (preds == labels)
            seq_is_correct = is_correct.sum(-1) == loss_counts
            
            pixel_similarity = is_correct.sum(-1).float() / loss_counts.float()
            
            valid_metrics = carry.halted & (loss_counts > 0)
            metrics = {
                "count": valid_metrics.sum(),
                "exact_accuracy": (valid_metrics & seq_is_correct).sum(),
                "steps": torch.where(valid_metrics, carry.steps, 0).sum(),
                "similarity": torch.where(valid_metrics, pixel_similarity, 0).sum(),
            }

        q_halt_loss = F.binary_cross_entropy_with_logits(q_halt_logits, seq_is_correct.to(q_halt_logits.dtype), reduction="sum")
        
        # BUG FIX: q_continue_loss가 계산되지 않는 문제 수정
        # target_q_continue가 명시적으로 주어지지 않으면, 아직 중단(halt)되지 않은 샘플에 대해
        # '계속(continue)'하도록 학습하는 것을 기본 동작으로 설정합니다.
        # 이상적인 '계속'의 타겟은 '~carry.halted' (중단되지 않음) 입니다.
        if target_q_continue is not None:
            # 외부에서 타겟을 제공한 경우
            q_continue_loss = F.binary_cross_entropy_with_logits(q_continue_logits, target_q_continue, reduction="sum")
        else:
            # 기본 동작: 아직 중단되지 않은 시퀀스는 계속하도록 학습
            # `carry.halted`는 bool 타입이므로 float 타입으로 변환해야 함
            continue_target = (~carry.halted).to(q_continue_logits.dtype)
            q_continue_loss = F.binary_cross_entropy_with_logits(q_continue_logits, continue_target, reduction="sum")

        metrics.update({
            "lm_loss": total_lm_loss.detach(),
            "q_halt_loss": q_halt_loss.detach(),
            "q_continue_loss": q_continue_loss.detach(),
        })

        final_loss = total_lm_loss + 0.5 * (q_halt_loss + q_continue_loss)

        return final_loss, metrics, lm_loss_per_seq