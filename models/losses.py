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


class ACTLossHead(nn.Module):
    def __init__(self, model: nn.Module, loss_type: str):
        super().__init__()
        self.model = model
        self.loss_fn = globals()[loss_type]
        
    def initial_carry(self, *args, **kwargs):
        return self.model.initial_carry(*args, **kwargs)  # type: ignore

    def forward(
        self,
        return_keys: Sequence[str],
        # Model args
        **model_kwargs,
    ) -> Tuple[Any, torch.Tensor, Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]], torch.Tensor]:
        # Model logits
        # B x SeqLen x D
        new_carry, outputs = self.model(**model_kwargs)
        labels = new_carry.current_data["labels"]

        # Correctness
        with torch.no_grad():
            mask = labels != IGNORE_LABEL_ID
            loss_counts = mask.sum(-1)
            loss_divisor = loss_counts.clamp_min(1).unsqueeze(-1)  # Avoid NaNs in division

            is_correct = mask & (torch.argmax(outputs["logits"], dim=-1) == labels)
            seq_is_correct = is_correct.sum(-1) == loss_counts
            
            # Metrics (halted)
            valid_metrics = new_carry.halted & (loss_counts > 0)
            metrics = {
                "count": valid_metrics.sum(),
                
                "accuracy":       torch.where(valid_metrics, (is_correct.to(torch.float32) / loss_divisor).sum(-1), 0).sum(),
                "exact_accuracy": (valid_metrics & seq_is_correct).sum(),

                "q_halt_accuracy": (valid_metrics & ((outputs["q_halt_logits"] >= 0) == seq_is_correct)).sum(),
                "steps":          torch.where(valid_metrics, new_carry.steps, 0).sum(),
            }

        # Losses
        # FIXME: Assuming the batch is always full
        lm_loss = (self.loss_fn(outputs["logits"], labels, ignore_index=IGNORE_LABEL_ID) / loss_divisor).sum()
        q_halt_loss = F.binary_cross_entropy_with_logits(outputs["q_halt_logits"], seq_is_correct.to(outputs["q_halt_logits"].dtype), reduction="sum")

        metrics.update({
            "lm_loss": lm_loss.detach(),
            "q_halt_loss": q_halt_loss.detach(),
        })

        # Q continue (bootstrapping target loss)
        q_continue_loss = 0
        if "target_q_continue" in outputs:
            q_continue_loss = F.binary_cross_entropy_with_logits(outputs["q_continue_logits"], outputs["target_q_continue"], reduction="sum")

            metrics["q_continue_loss"] = q_continue_loss.detach()

        # Filter outputs for return
        detached_outputs = {k: outputs[k].detach() for k in return_keys if k in outputs}

        return new_carry, lm_loss + 0.5 * (q_halt_loss + q_continue_loss), metrics, detached_outputs, new_carry.halted.all()


# +++ ARM을 위한 새로운 Loss Head 클래스 추가 +++
class ARMLossHead(nn.Module):
    def __init__(self, model: nn.Module, loss_type: str):
        super().__init__()
        self.model = model
        self.loss_fn = globals()[loss_type]
    
    def initial_carry(self, *args, **kwargs):
        # ARM 모델의 initial_carry를 호출합니다.
        return self.model.initial_carry(*args, **kwargs)

    def forward(
        self,
        # 이 Loss Head는 pretrain.py의 커스텀 훈련 루프에 의해 직접 제어됩니다.
        # 따라서 기존 ACTLossHead와 인터페이스가 다릅니다.
        carry: Any,
        batch: Dict[str, torch.Tensor],
        # 외부에서 계산된 값들을 직접 받습니다.
        final_logits: List[torch.Tensor],
        q_halt_logits: torch.Tensor,
        q_continue_logits: torch.Tensor,
        target_q_continue: Optional[torch.Tensor] = None
    ) -> Tuple[Any, torch.Tensor, Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]], torch.Tensor]:
        
        labels = carry.current_data["labels"]
        total_lm_loss = 0

        # --- 1. LM Loss 계산 ---
        # 모든 모듈의 최종 출력(logits)에 대해 LM Loss를 계산하여 합산합니다.
        for logits in final_logits:
            mask = labels != IGNORE_LABEL_ID
            loss_counts = mask.sum(-1)
            loss_divisor = loss_counts.clamp_min(1).unsqueeze(-1)
            
            # 각 시퀀스별 손실을 계산하고, 멈춘(halted) 시퀀스에 대해서만 전체 손실에 더합니다.
            lm_loss_per_seq = (self.loss_fn(logits, labels, ignore_index=IGNORE_LABEL_ID) / loss_divisor).sum(-1)
            total_lm_loss += torch.where(carry.halted, lm_loss_per_seq, 0).sum()

        # --- 2. 정확도 및 Q-러닝 관련 메트릭 계산 (ACTLossHead와 유사) ---
        with torch.no_grad():
            # 여기서는 마지막 모듈(활성 모듈)의 출력만을 기준으로 정확도를 계산합니다.
            last_logits = final_logits[0]
            mask = labels != IGNORE_LABEL_ID
            loss_counts = mask.sum(-1)
            loss_divisor = loss_counts.clamp_min(1).unsqueeze(-1)

            is_correct = mask & (torch.argmax(last_logits, dim=-1) == labels)
            seq_is_correct = is_correct.sum(-1) == loss_counts
            
            valid_metrics = carry.halted & (loss_counts > 0)
            metrics = {
                "count": valid_metrics.sum(),
                "accuracy": torch.where(valid_metrics, (is_correct.to(torch.float32) / loss_divisor).sum(-1), 0).sum(),
                "exact_accuracy": (valid_metrics & seq_is_correct).sum(),
                "q_halt_accuracy": (valid_metrics & ((q_halt_logits >= 0) == seq_is_correct)).sum(),
                "steps": torch.where(valid_metrics, carry.steps, 0).sum(),
            }

        # --- 3. Q-러닝 손실 계산 (ACTLossHead와 동일) ---
        q_halt_loss = F.binary_cross_entropy_with_logits(q_halt_logits, seq_is_correct.to(q_halt_logits.dtype), reduction="sum")

        metrics.update({
            "lm_loss": total_lm_loss.detach(),
            "q_halt_loss": q_halt_loss.detach(),
        })

        q_continue_loss = 0
        if target_q_continue is not None:
            q_continue_loss = F.binary_cross_entropy_with_logits(q_continue_logits, target_q_continue, reduction="sum")
            metrics["q_continue_loss"] = q_continue_loss.detach()

        # pretrain.py에서 gating_loss가 별도로 추가될 것입니다.
        # 여기서는 LM Loss와 Q-Loss만 합산합니다.
        final_loss = total_lm_loss + 0.5 * (q_halt_loss + q_continue_loss)

        # 출력할 detached_outputs은 비워둡니다. (평가 시 필요한 경우 pretrain.py에서 직접 처리)
        detached_outputs = {}

        return carry, final_loss, metrics, detached_outputs, carry.halted.all()
# from typing import Any, Tuple, Dict, Sequence, Optional

# import torch
# import torch.nn.functional as F
# from torch import nn


# IGNORE_LABEL_ID = -100


# def s(x, epsilon=1e-30):
#     return torch.where(
#         x<0,
#         1/(1-x+ epsilon),
#         x + 1
#     )


# def log_stablemax(x, dim=-1):
#     s_x = s(x)
#     return torch.log(s_x/torch.sum(s_x, dim=dim, keepdim=True))


# def stablemax_cross_entropy(logits, labels, ignore_index: int = -100):
#     logprobs = log_stablemax(logits.to(torch.float64), dim=-1)

#     valid_mask = labels != ignore_index
#     transformed_labels = torch.where(valid_mask, labels, 0)
#     prediction_logprobs = torch.gather(logprobs, index=transformed_labels.to(torch.long).unsqueeze(-1), dim=-1).squeeze(-1)

#     return -torch.where(valid_mask, prediction_logprobs, 0)


# def softmax_cross_entropy(logits, labels, ignore_index: int = -100):
#     # Cast logits to f32
#     # Flatten logits
#     return F.cross_entropy(logits.to(torch.float32).view(-1, logits.shape[-1]), labels.to(torch.long).view(-1), ignore_index=ignore_index, reduction="none").view(labels.shape)


# class ACTLossHead(nn.Module):
#     def __init__(self, model: nn.Module, loss_type: str):
#         super().__init__()
#         self.model = model
#         self.loss_fn = globals()[loss_type] #동적 함수 할당 코드
        
#     def initial_carry(self, *args, **kwargs):
#         return self.model.initial_carry(*args, **kwargs)  # type: ignore

#     def forward(
#         self,
#         return_keys: Sequence[str],
#         # Model args
#         **model_kwargs,
#     ) -> Tuple[Any, torch.Tensor, Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]], torch.Tensor]:
#         # Model logits
#         # B x SeqLen x D
#         new_carry, outputs = self.model(**model_kwargs)
#         labels = new_carry.current_data["labels"]

#         # Correctness
#         with torch.no_grad():
#             mask = labels != IGNORE_LABEL_ID
#             loss_counts = mask.sum(-1)
#             loss_divisor = loss_counts.clamp_min(1).unsqueeze(-1)  # Avoid NaNs in division

#             is_correct = mask & (torch.argmax(outputs["logits"], dim=-1) == labels)
#             seq_is_correct = is_correct.sum(-1) == loss_counts
            
#             # Metrics (halted)
#             valid_metrics = new_carry.halted & (loss_counts > 0)
#             metrics = {
#                 "count": valid_metrics.sum(),
                
#                 "accuracy":       torch.where(valid_metrics, (is_correct.to(torch.float32) / loss_divisor).sum(-1), 0).sum(),
#                 "exact_accuracy": (valid_metrics & seq_is_correct).sum(),

#                 "q_halt_accuracy": (valid_metrics & ((outputs["q_halt_logits"] >= 0) == seq_is_correct)).sum(),
#                 "steps":          torch.where(valid_metrics, new_carry.steps, 0).sum(),
#             }

#         # Losses
#         # FIXME: Assuming the batch is always full
#         lm_loss = (self.loss_fn(outputs["logits"], labels, ignore_index=IGNORE_LABEL_ID) / loss_divisor).sum()
#         q_halt_loss = F.binary_cross_entropy_with_logits(outputs["q_halt_logits"], seq_is_correct.to(outputs["q_halt_logits"].dtype), reduction="sum")

#         metrics.update({
#             "lm_loss": lm_loss.detach(),
#             "q_halt_loss": q_halt_loss.detach(),
#         })

#         # Q continue (bootstrapping target loss)
#         q_continue_loss = 0
#         if "target_q_continue" in outputs:
#             q_continue_loss = F.binary_cross_entropy_with_logits(outputs["q_continue_logits"], outputs["target_q_continue"], reduction="sum")

#             metrics["q_continue_loss"] = q_continue_loss.detach()

#         # Filter outputs for return
#         detached_outputs = {k: outputs[k].detach() for k in return_keys if k in outputs}

#         return new_carry, lm_loss + 0.5 * (q_halt_loss + q_continue_loss), metrics, detached_outputs, new_carry.halted.all()
