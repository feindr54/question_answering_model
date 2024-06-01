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
        self.decoder = GPT2LMHeadModel.from_pretrained("openai-community/gpt2").cuda(device)
        self.decoder.resize_token_embeddings(50257+2)
        print(self.decoder)
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

        print(long_answer_embeddings.shape)
        print(prompt_embeddings.shape)


        # concatenate the long answer embeddings to the prompts and labels
        for i in range(batch_size):
            row = prompt_embeddings[i]
            if indices[i] == -1:
                print("there is no long answer embedding space available")
            else:
                # Replace the special token with the long answer embeddings
                print(row[indices[i]].shape)
                print(long_answer_embeddings[3*i, i].shape)
                row[indices[i], :] = long_answer_embeddings[3*i, i] # prompt_embeddings qbatch x seqlen x 768 , long answer embedding abatch x qbatch x 768
        # obtain the loss from the decoder
        print("prompt_embedding shape: ", prompt_embeddings.shape)
        print("prompt mask shape: ", prompt_masks.shape)
        print("label shape: ", labels.shape)
        print(f"label_min={labels.min()}, label_max={labels.max()}")
        print(f"model shape={self.decoder}")
        outputs = self.decoder.forward(inputs_embeds=prompt_embeddings, attention_mask=prompt_masks, labels=labels)
        return outputs.loss

if __name__ == "__main__":
    from data import NQDataset, NQDataLoader
    from conf import device
    ds = NQDataset()
    dl = NQDataLoader(ds, batch_size=2)
    lae = torch.randn(size=(6,2,768))
    print(lae.shape)
    model = ShortAnswerDecoder(device)
    for batch in dl:
        loss = model(lae.cuda(device), batch["prompts"].cuda(device), batch['prompt_mask'].cuda(device), batch["short_answers_labels"].cuda(device))
        print(f"loss={loss}")
        break