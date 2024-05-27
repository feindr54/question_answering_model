import torch.nn as nn

from transformers import BertModel

class QuestionEncoder(nn.Module):
    def __init__(self, device) -> None:
        super(QuestionEncoder, self).__init__()
        self.qbert = BertModel.from_pretrained("google-bert/bert-base-uncased").cuda(device)

    def forward(self, question, question_mask):
        q_embeds = self.qbert(question, question_mask).pooler_output#.to(device=self.device) # (abatch) * qbatch x 768
        return q_embeds