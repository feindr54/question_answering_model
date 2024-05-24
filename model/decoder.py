import torch
import torch.nn as nn

from transformers import GPT2LMHeadModel
from conf import batch_size

class ShortAnswerDecoder(nn.Module):
    """
    Constructor of the Short Answer Model
    """
    def __init__(self, device):
        super(ShortAnswerDecoder, self).__init__()
        self.device = device
        self.decoder = GPT2LMHeadModel.from_pretrained("openai-community/gpt2") #.cuda(device)
        self.decoder.resize_token_embeddings(50257+100)
    """
    Forward pass of the short answer model.

    Args:
    - long_answer_embeddings: obtained from the long answer model
    - prompt: prompts for the decoder
    - prompt_mask: attention mask of the prompts
    - labels: the labels to train the decoder

    Returns:
    - loss: float computed from the decoder
    """
    def forward(self, long_answer_embeddings, prompts, prompt_masks, labels):
        # obtain the indices of special sep token for each prompt (1D tensor)
        # TODO - special token id is 50257, find a way not to hardcode that number
        special_token = 50257
        matches = (prompts == special_token)
        indices = torch.argmax(matches.int(), dim=1)
        # sets any non-matches to -1
        indices[~matches.any(dim=1)] = -1

        print(indices)

        batch_size = len(prompts)

        # convert the input ids to input embeddings
        with torch.no_grad():
            prompt_embeddings = self.decoder.transformer.wte(prompts)

        # concatenate the long answer embeddings to the prompts and labels
        for i in range(batch_size):
            row = prompt_embeddings[i]
            print(f"i: {i}")
            print(indices)
            print(f"index: {indices[i]}")
            if indices[i] == -1:
                print("there is no long answer embedding space available")
            else:
                # Replace the special token with the long answer embeddings
                row[indices[i], :] = long_answer_embeddings[i]
        print(labels)
        # obtain the loss from the decoder
        outputs = self.decoder.forward(inputs_embeds=prompt_embeddings, attention_mask=prompt_masks, labels=labels)
        return outputs.loss

if __name__ == "__main__":
    from data import NQDataset, NQDataLoader
    from conf import device
    ds = NQDataset()
    dl = NQDataLoader(ds, batch_size=1)
    lae = torch.randn(size=(1,768))
    print(lae.shape)
    model = ShortAnswerDecoder(device)
    for batch in dl:
        loss = model(lae, batch["prompts"], batch['prompt_mask'], batch["short_answers_labels"])
        print(f"loss={loss}")
        break