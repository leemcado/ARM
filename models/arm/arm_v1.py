from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel

from models.common import trunc_normal_init_
from models.layers import rms_norm, SwiGLU, Attention, RotaryEmbedding, CosSin, CastedEmbedding, CastedLinear
# CastedSparseEmbedding은 더 이상 필요 없으므로 제거해도 됩니다.
# from models.sparse_embedding import CastedSparseEmbedding 

# --- 1. 설정 클래스 (ARMConfig) 수정 ---
class ARMConfig(BaseModel):
    batch_size: int
    seq_len: int
    # puzzle_emb_ndim을 선택적(Optional)으로 변경하고 기본값을 0으로 설정
    puzzle_emb_ndim: int = 0 
    num_puzzle_identifiers: int
    vocab_size: int
    initial_modules: int
    max_modules: int
    inner_loops: int
    reasoning_layers: int
    hidden_size: int
    expansion: float
    num_heads: int
    pos_encodings: str
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    halt_max_steps: int
    halt_exploration_prob: float
    forward_dtype: str = "bfloat16"

# --- ARMBlock, ARMReasoningModule (변경 없음) ---
class ARMBlock(nn.Module):
    def __init__(self, config: ARMConfig) -> None:
        super().__init__()
        self.self_attn = Attention(
            hidden_size=config.hidden_size,
            head_dim=config.hidden_size // config.num_heads,
            num_heads=config.num_heads,
            num_key_value_heads=config.num_heads,
            causal=False
        )
        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion,
        )
        self.norm_eps = config.rms_norm_eps

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        # 1. 어텐션 출력(F(z))을 먼저 계산하고 정규화한 후, 입력(z)에 더합니다.
        attn_output = self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states)
        hidden_states = hidden_states + rms_norm(attn_output, variance_epsilon=self.norm_eps)

        # 2. MLP 출력(F(z))을 먼저 계산하고 정규화한 후, 입력(z)에 더합니다.
        mlp_output = self.mlp(hidden_states)
        hidden_states = hidden_states + rms_norm(mlp_output, variance_epsilon=self.norm_eps)
        return hidden_states


class ARMReasoningModule(nn.Module):
    def __init__(self, layers: List[ARMBlock]):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states, **kwargs)
        return hidden_states


@dataclass
class ARMInnerCarry:
    z_states: List[torch.Tensor]

# --- 3. ARM_Inner 클래스 수정 ---
class ARM_Inner(nn.Module):
    def __init__(self, config: ARMConfig) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, self.config.forward_dtype)

        self.embed_scale  = math.sqrt(self.config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale
        self.embed_tokens = CastedEmbedding(self.config.vocab_size, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        self.lm_head      = CastedLinear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.q_head       = CastedLinear(self.config.hidden_size, 2, bias=True)

        # --- puzzle_emb 생성 로직 제거 ---
        # self.puzzle_emb_len과 self.puzzle_emb를 조건 없이 제거합니다.
        
        if self.config.pos_encodings == "rope":
            # RoPE의 max_position_embeddings에서 puzzle_emb_len을 제거합니다.
            self.rotary_emb = RotaryEmbedding(dim=self.config.hidden_size // self.config.num_heads,
                                              max_position_embeddings=self.config.seq_len,
                                              base=self.config.rope_theta)
        elif self.config.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(self.config.seq_len, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        else:
            raise NotImplementedError()

        self.reasoning_modules = nn.ModuleList(
            [ARMReasoningModule(layers=[ARMBlock(self.config) for _ in range(self.config.reasoning_layers)]) 
             for _ in range(self.config.initial_modules)]
        )

        self.thalamus_gate = nn.Sequential(
            CastedLinear(self.config.hidden_size, self.config.hidden_size, bias=True),
            nn.ReLU(),
            CastedLinear(self.config.hidden_size, self.config.max_modules, bias=True)
        )

    # --- _input_embeddings 메소드 수정 ---
    def _input_embeddings(self, input: torch.Tensor, puzzle_identifiers: torch.Tensor):
        # Token embedding
        embedding = self.embed_tokens(input.to(torch.int32))

        # --- Puzzle embedding 관련 로직 전체 제거 ---
        # if self.config.puzzle_emb_ndim > 0: ... 블록을 전부 삭제합니다.

        if self.config.pos_encodings == "learned":
            embedding = 0.707106781 * (embedding + self.embed_pos.embedding_weight.to(self.forward_dtype))

        return self.embed_scale * embedding
        
    def add_new_expert_module(self):
        # ... (이 함수는 변경 없음) ...
        if len(self.reasoning_modules) < self.config.max_modules:
            new_module = ARMReasoningModule(
                layers=[ARMBlock(self.config) for _ in range(self.config.reasoning_layers)]
            )
            with torch.no_grad():
                for param in new_module.parameters():
                    trunc_normal_init_(param, std=1e-5)
            device = next(self.parameters()).device
            self.reasoning_modules.append(new_module.to(device))
            print(f"New expert module added. Total modules: {len(self.reasoning_modules)}")
            return True
        return False

    def empty_carry(self, batch_size: int):
        # ... (이 함수는 변경 없음) ...
        z_states = [torch.zeros(batch_size, self.config.seq_len, self.config.hidden_size, dtype=self.forward_dtype) 
                    for _ in range(len(self.reasoning_modules))]
        return ARMInnerCarry(z_states=z_states)
        
    def reset_carry(self, reset_flag: torch.Tensor, carry: ARMInnerCarry):
        # ... (이 함수는 변경 없음) ...
        new_z_states = []
        for z in carry.z_states:
            new_z_states.append(torch.where(reset_flag.view(-1, 1, 1), 0.0, z))
        return ARMInnerCarry(z_states=new_z_states)

    def forward(self, carry: ARMInnerCarry, batch: Dict[str, torch.Tensor]) -> Tuple[ARMInnerCarry, Dict[str, torch.Tensor]]:
        raise NotImplementedError("ARM_Inner.forward is controlled by the training loop in pretrain.py")

# --- 4. ACT 래퍼 (ARM) 수정 ---
@dataclass
class ARMCarry:
    inner_carry: ARMInnerCarry
    steps: torch.Tensor
    halted: torch.Tensor
    current_data: Dict[str, torch.Tensor]

class ARM(nn.Module):
    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = ARMConfig(**config_dict)
        self.inner = ARM_Inner(self.config)

    # --- puzzle_emb property 제거 ---
    # @property
    # def puzzle_emb(self):
    #     return self.inner.puzzle_emb

    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        # ... (이 함수는 변경 없음) ...
        batch_size = batch["inputs"].shape[0]
        return ARMCarry(
            inner_carry=self.inner.empty_carry(batch_size),
            steps=torch.zeros((batch_size, ), dtype=torch.int32),
            halted=torch.ones((batch_size, ), dtype=torch.bool),
            current_data={k: torch.empty_like(v) for k, v in batch.items()}
        )
        
    def forward(self, carry: ARMCarry, batch: Dict[str, torch.Tensor]) -> Tuple[ARMCarry, Dict[str, torch.Tensor]]:
        raise NotImplementedError("ARM.forward logic is handled by the custom training loop in pretrain.py")