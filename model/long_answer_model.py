import torch.nn as nn

from .question_encoder import QuestionEncoder
from .long_answer_encoder import LongAnswerEncoder
from .cross_attention import CrossAttentionModule

class LongAnswerModel(nn.Module):
    def __init__(self, device):
        super(LongAnswerModel, self).__init__()
        self.device = device

        self.question_encoder = QuestionEncoder(device)
        self.long_answer_encoder = LongAnswerEncoder(device)
        self.cross_attention = CrossAttentionModule(device)
        self.linear = nn.Linear(768,1, device=device) # input is batch * answers * 768

    def forward(self, question, question_mask, answer, answer_mask):
        # obtain bert embeddings of question and answer with attention mask
        # get the pooled output for the questions, and the unpooled output from the answers
        q_embeds = self.question_encoder(question.cuda(), question_mask.cuda()) # (abatch) * qbatch x 768
        a_embeds = self.long_answer_encoder(answer.cuda(), answer_mask.cuda()) # abatch x seqlen x 768

        # extend the dimensions of the questions to the answer batch
        q_embeds = q_embeds.repeat(a_embeds.shape[0],1,1)

        query = q_embeds
        key = a_embeds
        value = a_embeds

        embeddings = self.cross_attention(query, key, value)
        logits = self.linear(embeddings).squeeze(-1)
        return logits, embeddings

if __name__ == "__main__":
    from data import NQDataset, NQDataLoader
    from conf import device
    from torch.nn import BCEWithLogitsLoss
    ds = NQDataset()
    dl = NQDataLoader(ds, batch_size=1)
    model = LongAnswerModel(device)
    criterion = BCEWithLogitsLoss()
    for batch in dl:
        logits, embeddings = model(batch["questions"], batch["question_mask"], batch["long_answers"], batch['long_answer_mask'])
        loss = criterion(logits, batch["long_answers_labels"].cuda(device))
        print(f"loss={loss}")
        break