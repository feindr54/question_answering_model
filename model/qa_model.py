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
        # print(qtokens.shape)
        # print(qmask.shape)
        # print(atokens.shape)
        # print(amask.shape)
        la_logits, long_answer_embeddings = self.long_answer_model(qtokens, qmask, atokens, amask)

        # run the forward pass of the short answer model
        prompts, prompt_mask, labels = sa_inputs
        sa_loss = self.short_answer_model(long_answer_embeddings, prompts, prompt_mask, labels)

        # print("print long answer logits after running decoder")
        # print(la_logits)

        return la_logits, sa_loss

if __name__ == "__main__":
    from data import NQDataset, NQDataLoader
    from torch.nn import BCELoss as BCEWithLogitsLoss
    from conf import device
    data = NQDataset()
    dataloader = NQDataLoader(data, batch_size=2)
    model = QAModel(device)
    criterion = BCEWithLogitsLoss()
    for batch in dataloader:
        la_inputs = (batch["questions"], batch["question_mask"], batch["long_answers"], batch['long_answer_mask'])
        sa_inputs = (batch["prompts"].cuda(device), batch['prompt_mask'].cuda(device), batch["short_answers_labels"].cuda(device))
        la_logits, sa_loss = model(la_inputs, sa_inputs)
        print(la_logits)
        print(batch['long_answers_labels'])
        la_loss = criterion(la_logits, batch['long_answers_labels'])
        print(f"la_loss={la_loss}")
        print(f"sa_loss={sa_loss}")
        break
