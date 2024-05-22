import torch.nn as nn

from transformers import BertModel

class LongAnswerEncoder(nn.Module):
    def __init__(self, device) -> None:
        super(LongAnswerEncoder, self).__init__()
        self.abert = BertModel.from_pretrained("google-bert/bert-base-uncased").to(device)

    def forward(self, long_answer, long_answer_mask):
        a_embeds = self.qbert(long_answer, long_answer_mask).pooler_output.to(self.device) # (abatch) * qbatch x 768
        return a_embeds