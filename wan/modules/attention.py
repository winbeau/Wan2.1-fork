# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch

try:
    import flash_attn_interface

    def is_hopper_gpu():
        if not torch.cuda.is_available():
            return False
        device_name = torch.cuda.get_device_name(0).lower()
        return "h100" in device_name or "hopper" in device_name
    FLASH_ATTN_3_AVAILABLE = is_hopper_gpu()
except (ModuleNotFoundError, ImportError, OSError):
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except (ModuleNotFoundError, ImportError, OSError):
    FLASH_ATTN_2_AVAILABLE = False

# FLASH_ATTN_3_AVAILABLE = False

import warnings

__all__ = [
    'flash_attention',
    'attention',
    'attention_with_weights',
    'ATTENTION_WEIGHT_CAPTURE',
]


# Global attention weight capture configuration
class AttentionWeightCapture:
    """
    全局注意力权重捕获配置。

    重要：捕获的是 pre-softmax logits（注意力分数），而非 softmax 后的概率！
    这对于复现论文 Figure 4 至关重要，因为论文中的 Y 轴范围是 [-4, 6]。
    """
    def __init__(self):
        self.enabled = False
        self.layer_indices = None  # None 表示捕获所有层，或者是层索引列表
        self.captured_weights = []  # 捕获的注意力权重列表
        self.current_layer_idx = 0  # 当前前向传播的层索引
        self.capture_logits = True  # 是否捕获 pre-softmax logits（默认 True）
        self.num_layers = 30  # Wan 模型的层数，用于取模

    def enable(self, layer_indices=None, capture_logits=True, num_layers=30):
        """
        启用注意力权重捕获。

        Args:
            layer_indices: 要捕获的层索引列表，None 表示全部
            capture_logits: 如果 True，捕获 pre-softmax logits；否则捕获 post-softmax probs
            num_layers: 模型的总层数，用于 current_layer_idx 取模
        """
        self.enabled = True
        self.layer_indices = layer_indices
        self.capture_logits = capture_logits
        self.num_layers = num_layers
        self.captured_weights = []
        self.current_layer_idx = 0

    def disable(self):
        """禁用注意力权重捕获。"""
        self.enabled = False
        self.captured_weights = []
        self.current_layer_idx = 0

    def reset(self):
        """重置捕获的权重，用于新的前向传播。"""
        self.captured_weights = []
        self.current_layer_idx = 0

    def should_capture(self):
        """检查是否应该捕获当前层（使用模块化索引）。"""
        if not self.enabled:
            return False
        if self.layer_indices is None:
            return True
        # 使用模块化索引，这样每个 denoising step 的相同层都会被检查
        effective_layer_idx = self.current_layer_idx % self.num_layers
        return effective_layer_idx in self.layer_indices

    def get_effective_layer_idx(self):
        """获取当前的有效层索引（模块化后）。"""
        return self.current_layer_idx % self.num_layers

    def save(self, path):
        """保存捕获的权重到磁盘。"""
        import torch
        torch.save({
            'attention_weights': self.captured_weights,
            'layer_indices': self.layer_indices,
            'capture_logits': self.capture_logits,
        }, path)
        print(f"Saved attention weights to {path}")


ATTENTION_WEIGHT_CAPTURE = AttentionWeightCapture()


def flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    version=None,
):
    """
    q:              [B, Lq, Nq, C1].
    k:              [B, Lk, Nk, C1].
    v:              [B, Lk, Nk, C2]. Nq must be divisible by Nk.
    q_lens:         [B].
    k_lens:         [B].
    dropout_p:      float. Dropout probability.
    softmax_scale:  float. The scaling of QK^T before applying softmax.
    causal:         bool. Whether to apply causal attention mask.
    window_size:    (left right). If not (-1, -1), apply sliding window local attention.
    deterministic:  bool. If True, slightly slower and uses more memory.
    dtype:          torch.dtype. Apply when dtype of q/k/v is not float16/bfloat16.
    """
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type == 'cuda' and q.size(-1) <= 256

    # params
    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # preprocess query
    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = torch.tensor(
            [lq] * b, dtype=torch.int32).to(
                device=q.device, non_blocking=True)
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

    # preprocess key, value
    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.tensor(
            [lk] * b, dtype=torch.int32).to(
                device=k.device, non_blocking=True)
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale

    if version is not None and version == 3 and not FLASH_ATTN_3_AVAILABLE:
        warnings.warn(
            'Flash attention 3 is not available, use flash attention 2 instead.'
        )

    # apply attention
    if (version is None or version == 3) and FLASH_ATTN_3_AVAILABLE:
        # Note: dropout_p, window_size are not supported in FA3 now.
        x = flash_attn_interface.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic)[0].unflatten(0, (b, lq))
    else:
        assert FLASH_ATTN_2_AVAILABLE
        x = flash_attn.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic).unflatten(0, (b, lq))

    # output
    return x.type(out_dtype)


def attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    fa_version=None,
):
    # 检查是否需要捕获注意力权重
    if ATTENTION_WEIGHT_CAPTURE.enabled and ATTENTION_WEIGHT_CAPTURE.should_capture():
        out, attn_data = attention_with_weights(
            q=q, k=k, v=v,
            q_lens=q_lens, k_lens=k_lens,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            window_size=window_size,
            dtype=dtype,
            return_logits=ATTENTION_WEIGHT_CAPTURE.capture_logits,  # 根据配置返回 logits 或 probs
        )
        # 存储注意力权重（移到 CPU 以节省 GPU 内存）
        ATTENTION_WEIGHT_CAPTURE.captured_weights.append({
            'layer_idx': ATTENTION_WEIGHT_CAPTURE.get_effective_layer_idx(),  # 使用模块化索引
            'attn_weights': attn_data.cpu(),
            'q_shape': q.shape,
            'k_shape': k.shape,
            'is_logits': ATTENTION_WEIGHT_CAPTURE.capture_logits,  # 标记是 logits 还是 probs
        })
        ATTENTION_WEIGHT_CAPTURE.current_layer_idx += 1
        return out

    ATTENTION_WEIGHT_CAPTURE.current_layer_idx += 1

    if FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE:
        return flash_attention(
            q=q,
            k=k,
            v=v,
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
            dtype=dtype,
            version=fa_version,
        )
    else:
        if q_lens is not None or k_lens is not None:
            warnings.warn(
                'Padding mask is disabled when using scaled_dot_product_attention. It can have a significant impact on performance.'
            )
        attn_mask = None

        q = q.transpose(1, 2).to(dtype)
        k = k.transpose(1, 2).to(dtype)
        v = v.transpose(1, 2).to(dtype)

        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=causal, dropout_p=dropout_p)

        out = out.transpose(1, 2).contiguous()
        return out


def attention_with_weights(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    dtype=torch.bfloat16,
    return_logits=True,
):
    """
    计算注意力并返回注意力权重。
    这比 flash attention 慢，但允许我们捕获注意力权重用于可视化。

    Args:
        q: Query 张量，形状 [B, Lq, Nq, C]
        k: Key 张量，形状 [B, Lk, Nk, C]
        v: Value 张量，形状 [B, Lk, Nk, C]
        return_logits: 如果 True，返回 pre-softmax logits（用于 Figure 4）；
                      否则返回 post-softmax 概率

    Returns:
        out: 输出张量，形状 [B, Lq, Nq, C]
        attn_data: 注意力数据，形状 [B, Nq, Lq, Lk]
                  如果 return_logits=True，这是 pre-softmax 分数（可以是负值）
                  如果 return_logits=False，这是 post-softmax 概率 [0,1]
    """
    out_dtype = q.dtype

    # q: [B, Lq, N, C] -> [B, N, Lq, C]
    # k: [B, Lk, N, C] -> [B, N, Lk, C]
    # v: [B, Lk, N, C] -> [B, N, Lk, C]
    q = q.transpose(1, 2).to(dtype)
    k = k.transpose(1, 2).to(dtype)
    v = v.transpose(1, 2).to(dtype)

    if q_scale is not None:
        q = q * q_scale

    # Support GQA/MQA: Q heads can be a multiple of K/V heads (Nq must be divisible by Nk).
    if q.shape[1] != k.shape[1]:
        n_q, n_k = q.shape[1], k.shape[1]
        if n_q % n_k != 0:
            raise ValueError(f"Nq must be divisible by Nk, got Nq={n_q}, Nk={n_k}")
        repeat_factor = n_q // n_k
        k = k.repeat_interleave(repeat_factor, dim=1)
        v = v.repeat_interleave(repeat_factor, dim=1)

    # 计算缩放因子
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5

    # 计算注意力分数（logits）: [B, N, Lq, Lk]
    attn_scores = torch.matmul(q, k.transpose(-2, -1)) * softmax_scale

    bsz, _n, lq, lk = attn_scores.shape

    # Apply key padding mask if provided (k_lens is [B]).
    q_valid = None
    if k_lens is not None:
        key_idx = torch.arange(lk, device=attn_scores.device).view(1, 1, 1, lk)
        key_valid = key_idx < k_lens.view(bsz, 1, 1, 1)
        attn_scores = attn_scores.masked_fill(~key_valid, float('-inf'))

    # Track query validity to avoid NaNs when a padded query would be fully masked.
    if q_lens is not None:
        q_idx = torch.arange(lq, device=attn_scores.device).view(1, 1, lq, 1)
        q_valid = q_idx < q_lens.view(bsz, 1, 1, 1)
        attn_scores = attn_scores.masked_fill(~q_valid, 0.0)

    # 对齐非方阵 Q/K：query i 对应 key i + (lk - lq)。
    # 对于 varlen（k_lens/q_lens）场景，flash-attn 使用每个样本的有效长度来计算 offset，
    # 否则 window/causal 的对齐会与快路径不一致。
    if (k_lens is not None) or (q_lens is not None):
        lk_eff = k_lens if k_lens is not None else torch.full((bsz,), lk, device=attn_scores.device, dtype=torch.long)
        lq_eff = q_lens if q_lens is not None else torch.full((bsz,), lq, device=attn_scores.device, dtype=torch.long)
        offset = (lk_eff - lq_eff).view(bsz, 1, 1)  # [B,1,1]
    else:
        offset = lk - lq  # scalar

    q_pos = torch.arange(lq, device=attn_scores.device).view(1, lq, 1)  # [1,Lq,1]
    k_pos = torch.arange(lk, device=attn_scores.device).view(1, 1, lk)  # [1,1,Lk]
    center = q_pos + offset  # scalar or [B,1,1] -> [B,Lq,1]

    # 如果需要 causal mask
    if causal:
        # Mask positions where key is "in the future" relative to the aligned center.
        causal_mask = k_pos > center  # [B,Lq,Lk] or [1,Lq,Lk]
        attn_scores = attn_scores.masked_fill(causal_mask.unsqueeze(1), float('-inf'))

    # Sliding window local attention (if enabled).
    # Semantics follow the same "offset" convention as the causal mask for non-square Q/K:
    # query position i is aligned to key position i + (lk - lq) (varlen uses per-sample offset).
    if window_size != (-1, -1):
        left, right = window_size
        if left < 0:
            left = lk
        if right < 0:
            right = lk
        lower = center - left
        upper = center + right
        window_mask = (k_pos < lower) | (k_pos > upper)  # [B,Lq,Lk] or [1,Lq,Lk]
        attn_scores = attn_scores.masked_fill(window_mask.unsqueeze(1), float('-inf'))

    # 计算注意力权重（概率）
    # 当 key padding mask + window mask（或 causal mask）导致某些 query 行被完全屏蔽时，
    # softmax(-inf, -inf, ...) 会产生 NaN；flash-attn 在这种情况下会输出 0。
    row_has_any_valid = torch.isfinite(attn_scores).any(dim=-1, keepdim=True)
    attn_scores_for_softmax = attn_scores.masked_fill(~row_has_any_valid, 0.0)
    attn_weights = torch.softmax(attn_scores_for_softmax, dim=-1)
    attn_weights = attn_weights.masked_fill(~row_has_any_valid, 0.0)

    # 应用 dropout
    if dropout_p > 0.:
        attn_weights = torch.nn.functional.dropout(attn_weights, p=dropout_p)

    # 计算输出: [B, N, Lq, C]
    out = torch.matmul(attn_weights, v)

    if q_valid is not None:
        out = out.masked_fill(~q_valid, 0.0)
        attn_weights = attn_weights.masked_fill(~q_valid, 0.0)

    # 转置回来: [B, N, Lq, C] -> [B, Lq, N, C]
    out = out.transpose(1, 2).contiguous().to(out_dtype)

    # 根据配置返回 logits 或 probs
    if return_logits:
        return out, attn_scores  # 返回 pre-softmax logits
    else:
        return out, attn_weights  # 返回 post-softmax probs
