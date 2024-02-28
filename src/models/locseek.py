import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from src.models.pooling import Attention1dPoolingHead
from transformers import AutoTokenizer, EsmModel

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x, cos, sin):
    cos = cos[:, :, : x.shape[-2], :]
    sin = sin[:, :, : x.shape[-2], :]

    return (x * cos) + (rotate_half(x) * sin)

class RotaryEmbedding(torch.nn.Module):
    """
    Rotary position embeddings based on those in
    [RoFormer](https://huggingface.co/docs/transformers/model_doc/roformer). Query and keys are transformed by rotation
    matrices which depend on their relative positions.
    """

    def __init__(self, dim: int):
        super().__init__()
        # Generate and save the inverse frequency buffer (non trainable)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim))
        inv_freq = inv_freq
        self.register_buffer("inv_freq", inv_freq)

        self._seq_len_cached = None
        self._cos_cached = None
        self._sin_cached = None

    def _update_cos_sin_tables(self, x, seq_dimension=2):
        seq_len = x.shape[seq_dimension]

        # Reset the tables if the sequence length has changed,
        # or if we're on a new device (possibly due to tracing for instance)
        if seq_len != self._seq_len_cached or self._cos_cached.device != x.device:
            self._seq_len_cached = seq_len
            t = torch.arange(x.shape[seq_dimension], device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)

            self._cos_cached = emb.cos()[None, None, :, :]
            self._sin_cached = emb.sin()[None, None, :, :]

        return self._cos_cached, self._sin_cached

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self._cos_cached, self._sin_cached = self._update_cos_sin_tables(k, seq_dimension=-2)

        return (
            apply_rotary_pos_emb(q, self._cos_cached, self._sin_cached),
            apply_rotary_pos_emb(k, self._cos_cached, self._sin_cached),
        )


class CrossModalAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size  # 确保有hidden_size属性
        self.query_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.key_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.value_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.rotary_embeddings = RotaryEmbedding(dim=self.hidden_size // 2)  # 注意调整维度以适配RotaryEmbedding

    def forward(self, query, key, value, attention_mask=None):
        # 获取旋转位置编码
        cos, sin = self.rotary_embeddings._update_cos_sin_tables(query)

        # 应用旋转位置编码到query和key
        query_rot = apply_rotary_pos_emb(query, cos, sin)
        key_rot = apply_rotary_pos_emb(key, cos, sin)

        # 对经过旋转编码的query和key进行线性变换
        query = self.query_proj(query_rot)
        key = self.key_proj(key_rot)
        value = self.value_proj(value)
        
        # 计算注意力分数
        attn_scores = torch.matmul(query, key.transpose(-2, -1))
        attn_scores = attn_scores / (query.size(-1) ** 0.5)
        
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = (1.0 - attention_mask) * -10000.0
            attn_scores = attn_scores + attention_mask
        
        attn_probs = F.softmax(attn_scores, dim=-1)
        context = torch.matmul(attn_probs, value)
        attended = self.out_proj(context)
        return attended

class LocSeekModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.esm_model_name)
        self.model = EsmModel.from_pretrained(config.esm_model_name)
        self.pooling_method = Attention1dPoolingHead(config)
        self.foldseek_embedding = nn.Embedding(20, config.hidden_size)
        self.ss_embedding = nn.Embedding(8, config.hidden_size)
        
        self.cross_attention_ss = CrossModalAttention(config)
        self.cross_attention_foldseek = CrossModalAttention(config)

    def forward(self, batch):
        aa_seq, ss_seq, foldseek_seq = batch
        
        with torch.no_grad():
            inputs = self.tokenizer(aa_seq, return_tensors="pt", padding=True, truncation=True)
            input_ids = inputs["input_ids"].cuda()
            attention_mask = inputs["attention_mask"].cuda()
            outputs = self.model(input_ids, attention_mask=attention_mask)
            seq_embeds = outputs.last_hidden_state

        ss_embeds = self.ss_embedding(ss_seq)
        foldseek_embeds = self.foldseek_embedding(foldseek_seq)
        
        # 使用氨基酸序列信息引导二级结构信息的理解
        ss_context = self.cross_attention_ss(seq_embeds, ss_embeds, ss_embeds, attention_mask)
        
        # 结合二级结构和FoldSeek信息
        foldseek_context = self.cross_attention_foldseek(ss_context, foldseek_embeds, foldseek_embeds, attention_mask)
        
        # 将氨基酸序列、二级结构上下文、FoldSeek上下文聚合
        combined_embeds = torch.cat([seq_embeds, ss_context, foldseek_context], dim=1)
        final_embeds = self.pooling_method(combined_embeds, attention_mask)
        
        return final_embeds
       