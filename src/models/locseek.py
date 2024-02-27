import torch
import torch.nn as nn
from src.models.pooling import Attention1dPoolingHead
from transformers import AutoTokenizer, EsmModel


class LocSeekModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.esm_model_name)
        self.model = EsmModel.from_pretrained(self.config.esm_model_name).cuda()
        self.pooling_method = Attention1dPoolingHead(self.config)
        self.foldseek_embedding = nn.Embedding(20, self.config.hidden_size)
        self.ss_embedding = nn.Embedding(8, self.config.hidden_size)
        # self.foldseek_self_attention = nn.MultiheadAttention(self.config.hidden_size, 4)
        # self.ss_self_attention = nn.MultiheadAttention(self.config.hidden_size, 4)
        
        
    def forward(self, batch):
        aa_seq, ss_seq, foldseek_seq = batch
        with torch.no_grad():
            inputs = self.tokenizer(aa_seq, return_tensors="pt", padding=True, truncation=True)
            input_ids = inputs["input_ids"].cuda()
            attention_mask = inputs["attention_mask"].cuda()
            outputs = self.model(input_ids, attention_mask=attention_mask)
            seq_emebeds = outputs.last_hidden_state

        ss_seq = self.ss_embedding(ss_seq)
        foldseek_seq = self.foldseek_embedding(foldseek_seq)
        
        emebeds = torch.cat([seq_emebeds, ss_seq, foldseek_seq], dim=-1)
        logits = self.pooling_method(emebeds, attention_mask)
        
        return logits
       