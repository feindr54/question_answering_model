import torch.nn as nn

class CrossAttentionModule(nn.Module):
    def __init__(self, device, attention_layers=6):
        self.cx_attention = [] # output remains query size, ie abatch x seqlen x 768
        self.attention_layers = attention_layers

        for _ in range(self.attention_layers):
            self.cx_attention.append(nn.MultiheadAttention(768, 2, batch_first=True, device=device))
            self.cx_attention.append(nn.Linear(in_features=768, out_features=768, device=device))
            self.cx_attention.append(nn.ReLU())

    def forward(self, query, key, value):
        for i in range(self.attention_layers):
            # cross attention layer
            query = self.cx_attention[i*3](query, key, value)[0] # 0th index is attn output, 1st index is attention weights
            # feedforward layer
            query = self.cx_attention[i*3+2](self.cx_attention[i*3+1](query))
        return query
