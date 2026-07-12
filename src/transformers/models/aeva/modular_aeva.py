# Copyright 2026 the HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import itertools
import math
from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub.dataclasses import strict

from ... import initialization as init
from ...cache_utils import Cache, DynamicCache
from ...masking_utils import create_sliding_window_causal_mask
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_outputs import MoeCausalLMOutputWithPast, MoeModelOutputWithPast
from ...modeling_rope_utils import ROPE_INIT_FUNCTIONS
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring
from ..deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
from ..deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4Attention,
    DeepseekV4CSACache,
    DeepseekV4CSACompressor,
    DeepseekV4DecoderLayer,
    DeepseekV4Experts,
    DeepseekV4ForCausalLM,
    DeepseekV4GroupedLinear,
    DeepseekV4HashRouter,
    DeepseekV4HCACache,
    DeepseekV4HCACompressor,
    DeepseekV4HyperConnection,
    DeepseekV4HyperHead,
    DeepseekV4Indexer,
    DeepseekV4IndexerScorer,
    DeepseekV4Model,
    DeepseekV4PreTrainedModel,
    DeepseekV4RMSNorm,
    DeepseekV4RotaryEmbedding,
    DeepseekV4SparseMoeBlock,
    DeepseekV4TopKRouter,
    DeepseekV4UnweightedRMSNorm,
    apply_rotary_pos_emb,
    eager_attention_forward,
    load_balancing_loss_func,
)


@auto_docstring(checkpoint="louzongzi/Aeva-Nano")
@strict
class AevaConfig(DeepseekV4Config):
    r"""
    scoring_func (`str`):
        Router activation — `sqrtsoftplus`, `softmax`, or `sigmoid`.
    rope_theta (`float`):
        RoPE base for the main self-attention rotary.
    compress_rates (`dict[str, int]`):
        Per-layer-type compression rate.
    compress_rope_theta (`float`):
        RoPE base for the compressed branches.
    hc_mult (`int`):
        Hyper-Connection expansion factor.
    hc_eps (`float`):
        Numerical floor for Hyper-Connection normalization.
    mlp_layer_types (`list[str]`):
        Per-layer MoE schedule with values from `{"hash_moe", "moe"}`.
    swiglu_limit (`float`):
        Clip routed experts' gate/up pre-activations.
    sliding_window (`int`):
        Local window size for sliding-window attention.
    o_groups (`int`):
        Number of head-groups in the grouped output projection.
    o_lora_rank (`int`):
        Per-group intermediate dim in the grouped output projection.
    index_n_heads (`int`):
        Number of indexer query heads.
    index_head_dim (`int`):
        Indexer head dim.
    index_topk (`int`):
        Number of compressed entries per query via top-k.
    num_nextn_predict_layers (`int`):
        MTP layer count.
    partial_rotary_factor (`float`, *optional*):
        Fraction of head_dim that gets RoPE.
    block_size (`int`):
        Number of layers per block for Gated Delta Attention Residuals.
    moe_latent_size (`int`, *optional*):
        Latent size for MoE expert projections. If `None`, uses `hidden_size`.
    erc_loss_coef (`float`, *optional*):
        Weight coefficient for the expert-router coupling (ERC) loss.
    erc_alpha (`float`, *optional*):
        Scalar alpha hyperparameter for the ERC loss.
    """

    vocab_size: int = 154880
    hidden_size: int = 2048
    moe_intermediate_size: int = 2560
    num_hidden_layers: int = 27
    num_attention_heads: int = 32
    q_lora_rank: int = 512
    num_experts_per_tok: int = 4
    o_groups: int = 4
    n_shared_experts = AttributeError()
    hc_sinkhorn_iters = AttributeError()
    block_size: int = 6
    moe_latent_size: int | None = 512
    erc_loss_coef: float = 1.0
    erc_alpha: float = 0.5


class AevaRMSNorm(DeepseekV4RMSNorm):
    pass


class AevaUnweightedRMSNorm(DeepseekV4UnweightedRMSNorm):
    pass


class AevaRotaryEmbedding(DeepseekV4RotaryEmbedding):
    pass


class AevaHCACache(DeepseekV4HCACache):
    pass


class AevaCSACache(DeepseekV4CSACache):
    pass


class AevaGroupedLinear(DeepseekV4GroupedLinear):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_shape = x.shape[:-2]
        hidden_dim = x.shape[-1]
        w = self.weight.view(self.n_groups, -1, hidden_dim).transpose(1, 2)
        x = x.reshape(-1, self.n_groups, hidden_dim).transpose(0, 1)
        y = torch.bmm(x, w).transpose(0, 1)
        if self.bias is not None:
            y = y + self.bias.view(self.n_groups, -1)
        return y.reshape(*input_shape, self.n_groups, -1)


class AevaHCACompressor(DeepseekV4HCACompressor):
    pass


class AevaIndexerScorer(DeepseekV4IndexerScorer):
    pass


class AevaIndexer(DeepseekV4Indexer):
    pass


class AevaCSACompressor(DeepseekV4CSACompressor):
    pass


class AevaAttention(DeepseekV4Attention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: dict[str, tuple[torch.Tensor, torch.Tensor]] | tuple[torch.Tensor, torch.Tensor],
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        # position_embeddings is a {"main", "compress"} dict from the model; pick the
        # one that matches this layer's rope type (sliding → main, CSA/HCA → compress).
        cos, sin = position_embeddings[self.rope_layer_type]

        q_residual = self.q_a_norm(self.q_a_proj(hidden_states))
        q = self.q_b_proj(q_residual).view(*hidden_shape).transpose(1, 2)
        q = self.q_b_norm(q)
        q = apply_rotary_pos_emb(q, cos, sin)

        kv = self.kv_norm(self.kv_proj(hidden_states)).view(*hidden_shape)
        v_self = kv  # [B, S, 1, D] before transpose/rope, kept for orthogonal projection
        kv = kv.transpose(1, 2)
        kv = apply_rotary_pos_emb(kv, cos, sin)

        if past_key_values is not None:  # sliding where K==V
            kv = past_key_values.update(kv, kv, self.layer_idx)[0]

        block_bias = None
        if self.compressor is not None:  # Compressed KV (CSA or HCA)
            compressed_kv, block_bias = self.compressor(
                hidden_states, q_residual, position_ids, past_key_values, self.layer_idx
            )
            kv = torch.cat([kv, compressed_kv], dim=2)

        # The compressor path concatenates extra entries onto the KV axis after the
        # standard sliding-window cache update, so a tensor `attention_mask` (built
        # for the pre-concat KV length) needs to be extended to cover them. The
        # compressor returns a `block_bias` carrying per-query causality + indexer
        # validity over those new slots — cat it in instead of zero-padding (which
        # would let every query see every compressed slot).
        if isinstance(attention_mask, torch.Tensor) and kv.shape[2] > attention_mask.shape[-1]:
            if block_bias is not None:
                attention_mask = torch.cat([attention_mask, block_bias.to(attention_mask.dtype)], dim=-1)
            else:
                attention_mask = F.pad(attention_mask, (0, kv.shape[2] - attention_mask.shape[-1]), value=0.0)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )
        attn_output, attn_weights = attention_interface(
            self,
            q,
            kv,
            kv,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            s_aux=self.sinks,
            **kwargs,
        )

        # K=V in V4, so V picked up rope on its trailing rope slice. Apply the conjugate
        # rotation (`-sin`) at the query position to undo it on the rope slice of the
        # output before the grouped output projection mixes heads. The transpose pair is
        # just a layout fix-up: apply_rotary_pos_emb expects `[B, S, H, D]` (its
        # `unsqueeze_dim=1` adds a head-broadcast dim to cos/sin); attention gave us
        # `[B, H, S, D]`.
        attn_output = apply_rotary_pos_emb(attn_output.transpose(1, 2), cos, -sin).transpose(1, 2)

        # constrain attention to capture only information orthogonal to the token's own
        # value vector, encouraging better context modeling.
        # ref: https://arxiv.org/abs/2603.09078
        v_self = F.normalize(v_self, dim=-1)
        attn_output = attn_output - (attn_output * v_self).sum(dim=-1, keepdim=True) * v_self

        grouped = attn_output.reshape(*input_shape, self.config.o_groups, -1)
        grouped = self.o_a_proj(grouped).flatten(2)
        output = self.o_b_proj(grouped)
        return output, attn_weights


class AevaTopKRouter(DeepseekV4TopKRouter):
    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flat = hidden_states.reshape(-1, self.hidden_dim)
        # fp32 for numerical stability: entr(softmax(x)) involves p·log(p) which
        # blows up in low precision when p → 0.
        raw = F.linear(flat, self.weight).float()
        # Entropy-adaptive scaling: exp(2·H(p)/log(N) - 1) maps normalised entropy
        # H(p)/log(N) ∈ [0,1] to a multiplicative factor in [e⁻¹, e¹].
        # Low entropy (confident router) → scales down to prevent expert collapse;
        # high entropy (uncertain router) → scales up to reinforce exploration.
        entropy = torch.special.entr(F.softmax(raw, dim=-1)).sum(dim=-1, keepdim=True)
        scaling = torch.exp((2.0 * entropy / math.log(self.num_experts)) - 1.0)
        logits = raw * scaling
        scores = self.score_fn(logits)
        indices = torch.topk(scores + self.e_score_correction_bias.float(), self.top_k, dim=-1, sorted=False).indices
        weights = scores.gather(1, indices)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        return logits, (weights * self.routed_scaling_factor).to(hidden_states.dtype), indices


class AevaHashRouter(DeepseekV4HashRouter):
    pass


class AevaExperts(DeepseekV4Experts):
    def __init__(self, config: AevaConfig):
        super().__init__()
        self.hidden_dim = config.moe_latent_size


class AevaSparseMoeBlock(DeepseekV4SparseMoeBlock):
    def __init__(self, config: AevaConfig, layer_idx: int):
        super().__init__()
        del self.shared_experts
        self.fc1_latent_proj = nn.Linear(config.hidden_size, config.moe_latent_size, bias=config.mlp_bias)
        self.fc2_latent_proj = nn.Linear(config.moe_latent_size, config.hidden_size, bias=config.mlp_bias)

    def forward(self, hidden_states: torch.Tensor, input_ids: torch.Tensor | None = None) -> torch.Tensor:
        batch, seq_len, hidden_dim = hidden_states.shape
        _, weights, indices = self.gate(hidden_states, input_ids) if self.is_hash else self.gate(hidden_states)
        routed = self.experts(self.fc1_latent_proj(hidden_states.view(-1, hidden_dim)), indices, weights)
        return self.fc2_latent_proj(routed).view(batch, seq_len, hidden_dim)


class AevaHyperConnection(DeepseekV4HyperConnection):
    r"""mHC-lite from [mHC-lite: You Don't Need 20 Sinkhorn-Knopp Iterations](https://arxiv.org/abs/2601.05732) paper."""

    def __init__(self, config: AevaConfig):
        super().__init__()
        del self.hc_sinkhorn_iters
        self.input_norm = AevaUnweightedRMSNorm(eps=config.rms_norm_eps)
        mix = 2 * self.hc_mult + math.factorial(self.hc_mult)
        self.fn = nn.Parameter(torch.empty(mix, self.hc_mult * config.hidden_size))
        self.base = nn.Parameter(torch.empty(mix))
        self.register_buffer(
            "perm_mats",
            torch.stack([torch.eye(self.hc_mult)[list(p)] for p in itertools.permutations(range(self.hc_mult))]),
            persistent=False,
        )

    def forward(self, hidden_streams: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""
        Compute `pre`, `post`, `comb` from the mHC mapping (paper §2.2 eq. 8).
        `comb` is projected onto the doubly-stochastic manifold via Sinkhorn-
        Knopp: starting from the sigmoid-positive matrix, alternate row and
        column normalisation for `hc_sinkhorn_iters` steps. `pre` then collapses
        the `hc_mult` parallel streams into a single sequence (input projection
        into the sublayer); `post` and `comb` are returned for the caller to
        apply on the sublayer output.
        """
        hc, n_perms = self.hc_mult, self.perm_mats.shape[0]
        flat = self.input_norm(hidden_streams.flatten(start_dim=2).float())
        pre_w, post_w, comb_w = F.linear(flat, self.fn.float()).split([hc, hc, n_perms], dim=-1)
        pre_b, post_b, comb_b = self.base.split([hc, hc, n_perms])
        pre_scale, post_scale, comb_scale = self.scale.unbind(0)

        pre = torch.sigmoid(pre_w * pre_scale + pre_b) + self.hc_eps
        post = 2 * torch.sigmoid(post_w * post_scale + post_b)
        a = torch.softmax(comb_w * comb_scale + comb_b, dim=-1)
        comb = torch.einsum("bsp,pqk->bsqk", a, self.perm_mats.to(a.dtype))
        # Collapse the `hc_mult` parallel streams down to a single sequence using
        # the `pre` weights: one weighted sum across the stream axis, ready for
        # the sublayer (attn / MLP).
        collapsed = (pre.unsqueeze(-1) * hidden_streams).sum(dim=2).to(hidden_streams.dtype)
        return post, comb, collapsed


class AevaAttentionResidual(nn.Module):
    r"""Gated Delta Attention Residuals."""

    def __init__(self, config: AevaConfig):
        super().__init__()
        self.proj = AevaGroupedLinear(
            in_features_per_group=config.hidden_size,
            out_features=config.hc_mult * 4 * config.hidden_size,
            n_groups=config.hc_mult,
            bias=True,
        )
        self.norm = AevaUnweightedRMSNorm(config.rms_norm_eps)

    def forward(self, hidden_states: torch.Tensor, blocks: list[torch.Tensor]) -> torch.Tensor:
        B, S, H, D = hidden_states.shape

        values = torch.stack(blocks, dim=2).transpose(2, 3).reshape(-1, len(blocks), D)
        keys = self.norm(values)
        query, decay, erase, write = self.proj(hidden_states).split(D, dim=-1)

        # erase/write gates: erase modulates keys, write modulates values.
        # Ref: Gated DeltaNet-2 (https://arxiv.org/abs/2605.22791).
        queries = query.reshape(-1, 1, D)
        keys = keys * torch.sigmoid(erase.reshape(-1, 1, D))
        values = values * torch.sigmoid(write.reshape(-1, 1, D))

        aggregated = F.scaled_dot_product_attention(queries, keys, values, scale=1.0).view(B, S, H, D)
        return hidden_states * torch.sigmoid(-decay) + aggregated


def _is_block_start(layer_idx: int, n_hash: int, block_size: int) -> bool:
    """Whether *layer_idx* starts a new PathMoE router block."""
    if layer_idx == 0:
        return True
    if layer_idx < n_hash:
        return False
    if block_size <= 1:
        return True
    return (layer_idx - n_hash) % (block_size // 2) == 0


class AevaDecoderLayer(DeepseekV4DecoderLayer):
    r"""Aeva decoder block."""

    def __init__(self, config: AevaConfig, layer_idx: int):
        super().__init__()
        self.n_hash = config.default_num_hash_layers
        self.block_size = config.block_size
        self.attn_ar = AevaAttentionResidual(config)
        self.ffn_ar = AevaAttentionResidual(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        blocks: list[torch.Tensor] | None = None,
        partial_block: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor | None]:
        if blocks is None:
            blocks = [hidden_states]

        # block boundary: snapshot partial_block into blocks list.
        # Ref: https://arxiv.org/abs/2603.15031
        if (
            self.block_size > 1
            and partial_block is not None
            and _is_block_start(self.layer_idx, self.n_hash, self.block_size)
        ):
            blocks, partial_block = blocks + [partial_block], None

        hidden_states, partial_block, blocks = self._sublayer_step(
            hidden_states,
            blocks,
            partial_block,
            self.attn_ar,
            self.attn_hc,
            self.self_attn,
            self.input_layernorm,
            **kwargs,
        )
        hidden_states, partial_block, blocks = self._sublayer_step(
            hidden_states,
            blocks,
            partial_block,
            self.ffn_ar,
            self.ffn_hc,
            lambda x, **kw: self.mlp(x, input_ids=input_ids),
            self.post_attention_layernorm,
        )
        return hidden_states, blocks, partial_block

    def _sublayer_step(
        self,
        hidden_states: torch.Tensor,
        blocks: list[torch.Tensor],
        partial_block: torch.Tensor | None,
        ar: AevaAttentionResidual,
        hc: AevaHyperConnection,
        sublayer: Callable,
        norm: nn.Module,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[torch.Tensor]]:
        history = blocks + [partial_block] if partial_block is not None else blocks
        residual = ar(hidden_states, history)
        post, comb, collapsed = hc(residual)
        output = sublayer(norm(collapsed), **kwargs)
        if isinstance(output, tuple):
            output = output[0]
        dtype = hidden_states.dtype
        hidden_states = post.to(dtype).unsqueeze(-1) * output.unsqueeze(-2) + torch.matmul(
            comb.to(dtype).transpose(-1, -2), residual
        )
        delta = hidden_states - residual
        if self.block_size > 1:
            partial_block = delta if partial_block is None else partial_block + delta
        else:
            blocks = blocks + [delta]
        return hidden_states, partial_block, blocks


class AevaHyperHead(DeepseekV4HyperHead):
    pass


class AevaPreTrainedModel(DeepseekV4PreTrainedModel):
    def _init_weights(self, module):
        PreTrainedModel._init_weights(module)
        std = self.config.initializer_range
        d = self.config.hidden_size
        hc = self.config.hc_mult
        if isinstance(module, (AevaTopKRouter, AevaHashRouter)):
            init.normal_(module.weight, mean=0.0, std=std)
            if isinstance(module, AevaTopKRouter):
                init.zeros_(module.e_score_correction_bias)  # buffer
            if isinstance(module, AevaHashRouter):
                init.zeros_(module.tid2eid)  # buffer; real values come from the checkpoint
        elif isinstance(module, AevaExperts):
            init.normal_(module.gate_up_proj, mean=0.0, std=std)
            init.normal_(module.down_proj, mean=0.0, std=std)
        elif isinstance(module, AevaAttention):
            init.zeros_(module.sinks)
        elif isinstance(module, AevaHyperConnection):
            init.normal_(module.fn, mean=0.0, std=std)
            init.zeros_(module.base)
            init.ones_(module.scale)
        elif isinstance(module, AevaHyperHead):
            init.normal_(module.hc_fn, mean=0.0, std=std)
            init.zeros_(module.hc_base)
            init.ones_(module.hc_scale)
        elif isinstance(module, (AevaHCACompressor, AevaCSACompressor, AevaIndexer)):
            init.zeros_(module.position_bias)
        elif isinstance(module, AevaRotaryEmbedding):
            for layer_type in module.layer_types:
                rope_init_fn = module.compute_default_rope_parameters
                if module.rope_type[layer_type] != "default":
                    rope_init_fn = ROPE_INIT_FUNCTIONS[module.rope_type[layer_type]]
                curr_inv_freq, _ = rope_init_fn(module.config, layer_type=layer_type)
                init.copy_(getattr(module, f"{layer_type}_inv_freq"), curr_inv_freq)
                init.copy_(getattr(module, f"{layer_type}_original_inv_freq"), curr_inv_freq)
        elif isinstance(module, AevaAttentionResidual):
            init.zeros_(module.proj.weight)
            if module.proj.bias is not None:
                init.zeros_(module.proj.bias)
                module.proj.bias.view(hc, 4 * d)[:, 1 * d : 2 * d] = -10.0
                module.proj.bias.view(hc, 4 * d)[:, 3 * d : 4 * d] = -10.0


class AevaModel(DeepseekV4Model):
    def __init__(self, config: AevaConfig):
        super().__init__(config)
        self._tie_block_routers()

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> MoeModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if position_ids is None:
            past_seen = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen
            position_ids = position_ids.unsqueeze(0)
            # `generate()` may pass a per-layer-type mask dict already built by
            # `create_masks_for_generate`; all V4 layer types use the same sliding-window
            # mask, so use the prebuilt one directly. Otherwise build it here.
        if isinstance(attention_mask, dict):
            causal_mask = next(iter(attention_mask.values()))
        else:
            causal_mask = create_sliding_window_causal_mask(
                config=self.config,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
            )
        hidden_states = inputs_embeds.unsqueeze(2).expand(-1, -1, self.config.hc_mult, -1).contiguous()
        position_embeddings = {
            "main": self.rotary_emb(inputs_embeds, position_ids=position_ids, layer_type="main"),
            "compress": self.rotary_emb(inputs_embeds, position_ids=position_ids, layer_type="compress"),
        }

        blocks, partial_block = None, None
        for layer in self.layers:
            hidden_states, blocks, partial_block = layer(
                hidden_states,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                attention_mask=causal_mask,
                input_ids=input_ids,
                past_key_values=past_key_values,
                blocks=blocks,
                partial_block=partial_block,
                **kwargs,
            )

        hidden_states = self.norm(self.hc_head(hidden_states))
        return MoeModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=past_key_values)

    def _tie_block_routers(self):
        cfg, gate, gate_idx, tied = self.config, None, 0, {}
        for i, layer in enumerate(self.layers):
            if _is_block_start(i, cfg.default_num_hash_layers, cfg.block_size):
                gate, gate_idx = layer.mlp.gate, i
            else:
                layer.mlp.gate = gate
                tied[f"layers.{i}.mlp.gate.weight"] = f"layers.{gate_idx}.mlp.gate.weight"
                tied[f"layers.{i}.mlp.gate.e_score_correction_bias"] = (
                    f"layers.{gate_idx}.mlp.gate.e_score_correction_bias"
                )
        self._tied_weights_keys = tied


def erc_loss_func(
    router_weights: torch.Tensor | tuple[torch.Tensor] | None,
    expert_gate_weights: torch.Tensor | tuple[torch.Tensor] | None,
    alpha: float = 1.0,
    intermediate_dim: int | None = None,
) -> torch.Tensor | int:
    r"""
    Computes expert-router coupling (ERC) loss as in "Coupling Experts and Routers in Mixture-of-Experts via an Auxiliary Loss".

    See the paper (https://arxiv.org/abs/2512.23447) for more details. This function implements the loss
    function presented in Figure 1 and Equation (3) of the paper. It aims at tightly coupling the router's
    decisions with expert capabilities by treating router parameters as cluster centers.

    Args:
        router_weights:
            Weights from the `router`, should be a tuple of model.config.num_hidden_layers tensors of
            shape [num_experts, hidden_size].
        expert_gate_weights:
            Weights from the `expert_gate_proj` (or packed `gate_up_proj`), should be a tuple of tensors of
            shape [num_experts, 2 * intermediate_dim, hidden_size] or [num_experts, intermediate_dim, hidden_size].
        alpha (`float`, *optional*, defaults to 1.0):
            Scalar hyperparameter governing the coupling strength and expert specialization.
        intermediate_dim (`int`, *optional*):
            The intermediate size of the expert, used to slice the gate projection from packed `gate_up_proj` if necessary.

    Returns:
        The expert-router coupling loss.
    """
    total_erc_loss = 0.0

    for R, Wg in zip(router_weights, expert_gate_weights):
        if Wg.dim() == 3 and intermediate_dim is not None and Wg.shape[1] == 2 * intermediate_dim:
            Wg = Wg[:, :intermediate_dim, :]

        with torch.no_grad():
            R_fp32 = R.float()
            norm_R = torch.norm(R_fp32, dim=1)
            distances = torch.cdist(R_fp32, R_fp32, p=2)
            distances.fill_diagonal_(float("inf"))
            min_dist, _ = torch.min(distances, dim=1)
            eps = min_dist / 2 / norm_R

            low = (1 - eps).unsqueeze(1)
            high = (1 + eps).unsqueeze(1)
            noise = torch.rand_like(R_fp32)
            R = ((low + noise * (high - low)) * R_fp32).to(Wg.dtype)

        M = torch.norm(torch.einsum("jDd,id->ijD", Wg, R), dim=-1)

        row_diff = M - alpha * torch.diag(M).unsqueeze(1)
        row_diff_clamped = torch.clamp(row_diff, min=0.0)

        col_diff = M - alpha * torch.diag(M).unsqueeze(0)
        col_diff_clamped = torch.clamp(col_diff, min=0.0)

        mask = torch.ones_like(M) - torch.eye(M.size(0), device=M.device)
        total_diff = (row_diff_clamped + col_diff_clamped) * mask

        total_erc_loss += total_diff.mean()

    return total_erc_loss


class AevaForCausalLM(DeepseekV4ForCausalLM):
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_router_logits: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> MoeCausalLMOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Example:

        ```python
        >>> from transformers import AutoTokenizer, AevaForCausalLM

        >>> model = AevaForCausalLM.from_pretrained("louzongzhi/Aeva-Nano")
        >>> tokenizer = AutoTokenizer.from_pretrained("louzongzhi/Aeva-Nano")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs: MoeModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_router_logits=output_router_logits,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.vocab_size, **kwargs)

        aux_loss = None
        if output_router_logits:
            aux_loss = load_balancing_loss_func(
                outputs.router_logits,
                self.num_experts,
                self.num_experts_per_tok,
                attention_mask,
            )
            if labels is not None:
                loss += self.router_aux_loss_coef * aux_loss.to(loss.device)  # make sure to reside in the same device

        erc_loss = erc_loss_func(
            tuple(layer.mlp.fc1_latent_proj(layer.mlp.gate.weight) for layer in self.model.layers),
            tuple(layer.mlp.experts.gate_up_proj for layer in self.model.layers),
            alpha=self.config.erc_alpha,
            intermediate_dim=self.config.intermediate_size,
        )
        if labels is not None:
            loss += self.config.erc_loss_coef * erc_loss.to(loss.device)

        return MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
        )


__all__ = [
    "AevaConfig",
    "AevaPreTrainedModel",
    "AevaModel",
    "AevaForCausalLM",
]
