import torch.nn as nn

from .long_answer_model import LongAnswerModel
from .decoder import ShortAnswerDecoder

class QAModel(nn.Module):
    """
    Constructor of the QA model
    """
    def __init__(self, device):
        super(QAModel, self).__init__()
        # contains a long answer model and short answer model
        self.long_answer_model = LongAnswerModel(device)
        self.short_answer_model = ShortAnswerDecoder(device)
        self.device = device

    """
    Determines the forward function of the overall QA model

    Args:
    - la_inputs: long answer inputs, Tuple of
        - question tokens: 2D tensor
        - answer tokens: 2D tensor
        - question mask: 2D tensor
        - answer mask: 2D tensor
    - sa_inputs: short answer inputs, Tuple of (long answer ids obtained from la_inputs)
        - long_answer_embeddings: 2D tensor of embeddings from the long answer model
        - prompts: 2D tensor of tokens
        - prompt_mask: 2D tensor of tokens
        - labels: 2D tensor of tokens

    Returns:
    - losses: Tuple of
        - la_loss: resulting from the long answer model (encoder)
        - sa_loss: resulting from the short answer model (decoder)
    """
    def forward(self, la_inputs, sa_inputs):
        # run the forward pass to the long answer retriever, and retrieve a long answer embedding
        qtokens, atokens, qmask, amask = la_inputs
        la_logits, long_answer_embeddings = self.long_answer_model(qtokens, atokens, qmask, amask)

        # run the forward pass of the short answer model
        prompts, prompt_mask, labels = sa_inputs
        sa_loss = self.short_answer_model(long_answer_embeddings, prompts, prompt_mask, labels)

        return la_logits, sa_loss