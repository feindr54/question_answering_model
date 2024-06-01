import torch.nn as nn

from transformers import BertModel

class QuestionEncoder(nn.Module):
    def __init__(self, device) -> None:
        super(QuestionEncoder, self).__init__()
        self.qbert = BertModel.from_pretrained("google-bert/bert-base-uncased").cuda(device)

    def forward(self, question, question_mask):
        q_embeds = self.qbert(question, question_mask).pooler_output#.to(device=self.device) # (abatch) * qbatch x 768
        return q_embeds

if __name__ == "__main__":
    import torch
    from conf import device
    qencoder = QuestionEncoder(device=device)
    question_tokens = torch.randint(size=(30,10), low=1, high=20000).cuda()
    qmask = torch.full_like(question_tokens, fill_value=1, dtype=int).cuda()
    encoded = qencoder(question_tokens, qmask)
    print(encoded.shape)