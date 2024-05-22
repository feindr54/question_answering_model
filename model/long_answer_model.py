import torch.nn as nn

from question_encoder import QuestionEncoder
from long_answer_encoder import LongAnswerEncoder
from cross_attention import CrossAttentionModule

class LongAnswerModel(nn.Module):
    def __init__(self, device):
        super(LongAnswerModel, self).__init__()
        self.device = device

        self.question_encoder = QuestionEncoder(device)
        self.long_answer_encoder = LongAnswerEncoder(device)
        self.cross_attention = CrossAttentionModule(device)
        self.linear = nn.Linear(768,1, device=device) # input is batch * answers * 768

    def forward(self, question, answer, question_mask, answer_mask):
        # obtain bert embeddings of question and answer with attention mask
        # get the pooled output for the questions, and the unpooled output from the answers
        q_embeds = self.qbert(question, question_mask).pooler_output.to(self.device) # (abatch) * qbatch x 768
        a_embeds = self.abert(answer, answer_mask).last_hidden_state.to(self.device) # abatch x seqlen x 768

        # extend the dimensions of the questions to the answer batch
        q_embeds = q_embeds.repeat(a_embeds.shape[0],1,1)

        query = q_embeds
        key = a_embeds
        value = a_embeds

        output = self.cross_attention(query, key, value)
        output = self.linear(output)
        return output