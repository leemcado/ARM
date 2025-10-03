from typing import Tuple, List, Dict, Any
from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel

from models.common import trunc_normal_init_
from models.layers import rms_norm, SwiGLU, Attention, RotaryEmbedding, CosSin, CastedEmbedding, CastedLinear

# --- 1. 설정 클래스 (ARMConfig) 수정 ---
# YAML 파일의 중첩된 구조(frontal_module, thalamus_module 등)를 파싱하기 위한 Pydantic 모델들
class FrontalModuleConfig(BaseModel):
    num_layers: int
    hidden_size: int
    num_heads: int
    expansion: float

class ThalamusModuleConfig(BaseModel):
    hidden_size: int
    num_layers: int

class ReasoningModuleConfig(BaseModel):
    num_layers: int
    hidden_size: int
    num_heads: int
    expansion: float
    inner_loops: int

class ARMConfig(BaseModel):
    # Pydantic 모델이 YAML의 중첩 구조를 인식하도록 필드를 수정
    frontal_module: FrontalModuleConfig
    thalamus_module: ThalamusModuleConfig
    reasoning_module: ReasoningModuleConfig
    
    # 공통 및 기타 설정
    batch_size: int
    seq_len: int
    vocab_size: int
    num_puzzle_identifiers: int
    pos_encodings: str
    initial_modules: int
    max_modules: int
    
    # ACT 관련 설정
    halt_max_steps: int
    halt_exploration_prob: float

    # 기술적 세부사항
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    forward_dtype: str = "bfloat16"


# --- 2. 모듈별 클래스 정의 ---
# 각 모듈은 nn.Module을 상속받는 독립적인 클래스로 명확하게 분리

class TransformerBlock(nn.Module):
    """표준 트랜스포머 인코더 블록 (기존 ARMBlock과 동일)"""
    def __init__(self, hidden_size: int, num_heads: int, expansion: float, norm_eps: float):
        super().__init__()
        self.self_attn = Attention(
            hidden_size=hidden_size,
            head_dim=hidden_size // num_heads,
            num_heads=num_heads,
            num_key_value_heads=num_heads,
            causal=False
        )
        self.mlp = SwiGLU(
            hidden_size=hidden_size,
            expansion=expansion,
        )
        self.norm_eps = norm_eps

    def forward(self, hidden_states: torch.Tensor, cos_sin: CosSin) -> torch.Tensor:
        # Post-Normalization 구조
        hidden_states = rms_norm(hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states), variance_epsilon=self.norm_eps)
        hidden_states = rms_norm(hidden_states + self.mlp(hidden_states), variance_epsilon=self.norm_eps)
        return hidden_states

class FrontalModule(nn.Module):
    """전두엽 모듈: 트랜스포머 기반의 총괄 계획 및 의사결정자"""
    def __init__(self, config: FrontalModuleConfig, main_config: ARMConfig):
        super().__init__()
        self.layers = nn.ModuleList(
            [TransformerBlock(config.hidden_size, config.num_heads, config.expansion, main_config.rms_norm_eps) 
             for _ in range(config.num_layers)]
        )

    def forward(self, z_f: torch.Tensor, z_r_active: torch.Tensor, cos_sin: CosSin) -> torch.Tensor:
        # 전두엽 상태와 활성 추론 모듈의 출력을 합쳐서 다음 계획을 수립
        x = z_f + z_r_active
        for layer in self.layers:
            x = layer(x, cos_sin)
        return x

class ThalamusModule(nn.Module):
    """시상 모듈: ANN 기반의 문제 분류 및 라우터"""
    def __init__(self, config: ThalamusModuleConfig, main_config: ARMConfig):
        super().__init__()
        layers = [CastedLinear(main_config.reasoning_module.hidden_size, config.hidden_size, bias=True), nn.ReLU()]
        for _ in range(config.num_layers - 1):
            layers.extend([CastedLinear(config.hidden_size, config.hidden_size, bias=True), nn.ReLU()])
        layers.append(CastedLinear(config.hidden_size, main_config.max_modules, bias=True))
        self.net = nn.Sequential(*layers)

    def forward(self, z_f: torch.Tensor, input_embedding: torch.Tensor) -> torch.Tensor:
        # 전두엽 상태와 입력 문제의 첫 토큰을 보고 각 모듈의 예상 오류를 예측
        x = z_f[:, 0, :] + input_embedding[:, 0, :]
        return self.net(x)

class ReasoningModule(nn.Module):
    """추론 모듈: 트랜스포머 기반의 분야별 전문가"""
    def __init__(self, config: ReasoningModuleConfig, main_config: ARMConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [TransformerBlock(config.hidden_size, config.num_heads, config.expansion, main_config.rms_norm_eps) 
             for _ in range(config.num_layers)]
        )

    def forward(self, z_r: torch.Tensor, z_f: torch.Tensor, input_embedding: torch.Tensor, cos_sin: CosSin) -> torch.Tensor:
        # 자신의 상태, 전두엽 계획, 입력 문제를 모두 합쳐 내부 재귀 연산 수행
        x = z_r + z_f + input_embedding
        for _ in range(self.config.inner_loops):
            for layer in self.layers:
                x = layer(x, cos_sin)
        return x

# --- 3. Carry 객체 재정의 ---
@dataclass
class ARMInnerCarry:
    """시간에 따라 전달될 히든 스테이트들을 담는 객체"""
    z_f: torch.Tensor  # 전두엽 모듈의 히든 스테이트
    z_r_states: List[torch.Tensor] # 각 추론 모듈의 히든 스테이트 리스트

@dataclass
class ARMCarry:
    """ACT와 결합된 전체 Carry 객체"""
    inner_carry: ARMInnerCarry
    steps: torch.Tensor
    halted: torch.Tensor
    current_data: Dict[str, Any]

# --- 4. 메인 모델 (ARM_Inner, ARM) 재구성 ---
class ARM_Inner(nn.Module):
    """새로운 아키텍처의 모든 구성요소를 소유하고 관리하는 컨테이너"""
    def __init__(self, config: ARMConfig) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, config.forward_dtype)

        # --- 기본 I/O 레이어 (토큰 임베딩, 최종 출력) ---
        self.embed_scale  = math.sqrt(config.reasoning_module.hidden_size)
        embed_init_std = 1.0 / self.embed_scale
        self.embed_tokens = CastedEmbedding(config.vocab_size, config.reasoning_module.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        self.lm_head      = CastedLinear(config.frontal_module.hidden_size, config.vocab_size, bias=False)
        self.q_head       = CastedLinear(config.frontal_module.hidden_size, 2, bias=True)

        # --- 위치 인코딩 ---
        if self.config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(dim=config.reasoning_module.hidden_size // config.reasoning_module.num_heads,
                                              max_position_embeddings=config.seq_len,
                                              base=config.rope_theta)
        
        # --- 핵심 모듈 인스턴스화 ---
        self.frontal_module = FrontalModule(config.frontal_module, config)
        self.thalamus_module = ThalamusModule(config.thalamus_module, config)
        self.reasoning_modules = nn.ModuleList(
            [ReasoningModule(config.reasoning_module, config) for _ in range(config.initial_modules)]
        )

    def add_new_reasoning_module(self):
        """추론 모듈을 동적으로 추가하는 함수"""
        if len(self.reasoning_modules) < self.config.max_modules:
            device = next(self.parameters()).device
            new_module = ReasoningModule(self.config.reasoning_module, self.config).to(device)
            self.reasoning_modules.append(new_module)
            print(f"New reasoning module added. Total modules: {len(self.reasoning_modules)}")
            return True
        return False

    def _input_embeddings(self, input_ids: torch.Tensor):
        """입력 ID를 임베딩 벡터로 변환"""
        embedding = self.embed_tokens(input_ids.to(torch.int32))
        return self.embed_scale * embedding

    # 순전파 로직은 pretrain.py로 이전되었으므로, forward 메소드는 제거/단순화
    def forward(self, *args, **kwargs):
        raise NotImplementedError("ARM_Inner.forward is not meant to be called directly. Logic is in pretrain.py")

class ARM(nn.Module):
    """최상위 모델 래퍼. ACT와 관련된 상태를 관리"""
    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = ARMConfig(**config_dict)
        self.inner = ARM_Inner(self.config)

    def initial_carry(self, batch: Dict[str, torch.Tensor]) -> ARMCarry:
        """훈련 시작 시 초기 히든 스테이트를 생성"""
        batch_size = batch["inputs"].shape[0]
        
        # 전두엽과 초기 추론 모듈들의 히든 스테이트를 0으로 초기화
        z_f = torch.zeros(batch_size, self.config.seq_len, self.config.frontal_module.hidden_size, 
                          dtype=getattr(torch, self.config.forward_dtype), device="cuda")
        
        # BUG FIX: 초기 모듈 개수가 아닌, 현재 활성화된 실제 모듈 개수만큼 z_r_states를 생성
        num_current_modules = len(self.inner.reasoning_modules)
        z_r_states = [torch.zeros(batch_size, self.config.seq_len, self.config.reasoning_module.hidden_size, 
                                  dtype=getattr(torch, self.config.forward_dtype), device="cuda")
                      for _ in range(num_current_modules)]
        
        inner_carry = ARMInnerCarry(z_f=z_f, z_r_states=z_r_states)

        return ARMCarry(
            inner_carry=inner_carry,
            steps=torch.zeros((batch_size,), dtype=torch.int32, device="cuda"),
            halted=torch.ones((batch_size,), dtype=torch.bool, device="cuda"),
            current_data={k: v.clone() for k, v in batch.items()} # 초기 데이터를 저장
        )

    def forward(self, *args, **kwargs):
        raise NotImplementedError("ARM.forward logic is handled by the custom training loop in pretrain.py")