import torch.nn as nn

class CrossAttentionModule(nn.Module):
    def __init__(self, device, attention_layers=6):
        super(CrossAttentionModule, self).__init__()
        self.cx_attention = [] # output remains query size, ie abatch x seqlen x 768
        self.attention_layers = attention_layers

        for _ in range(self.attention_layers):
            self.cx_attention.append(nn.MultiheadAttention(768, 2, batch_first=True, device=device))
            self.cx_attention.append(nn.Linear(in_features=768, out_features=768, device=device))
            self.cx_attention.append(nn.ReLU())

    def forward(self, query, key, value):
        query = query.cuda()
        key = key.cuda()
        value = value.cuda()
        for i in range(self.attention_layers):
            # cross attention layer
            query = self.cx_attention[i*3](query, key, value)[0] # 0th index is attn output, 1st index is attention weights
            # feedforward layer
            query = self.cx_attention[i*3+2](self.cx_attention[i*3+1](query))
        return query

if __name__ == "__main__":
    import torch
    from conf import device
    cross_attention = CrossAttentionModule(device)
    query = torch.randint(size=(10, 768), low=0, high=20000).float()
    key = torch.randint(size=(10,768), low=0, high=20000).float()
    value = key
    embeddings = cross_attention(query, key, value)
    print(embeddings.shape)