import torch
import torch.nn as nn

from transformers import GPT2LMHeadModel
from conf import batch_size

class ShortAnswerDecoder(nn.Module):
    """
    Constructor of the Short Answer Model
    """
    def __init__(self, device):
        super(ShortAnswerDecoder, self).__init__(device=device)
        self.device = device
        self.decoder = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")

    """
    Forward pass of the short answer model.

    Args:
    - long_answer_ids: long answer input ids (tokens)
    - long_answer_mask: long answer attention masks

    """
    def forward(self, long_answer_embeddings, prompts, prompt_masks, labels):
        # obtain the indices of special sep token for each prompt (1D tensor)
        # TODO - special token id is 50257, find a way not to hardcode that number
        special_token = 50257
        matches = (prompts == special_token)
        indices = torch.argmax(matches.int(), dim=1)
        # sets any non-matches to -1
        indices[~matches.any(dim=1)] = -1

        # convert the input ids to input embeddings
        with torch.no_grad():
            prompt_embeddings = self.decoder.wte(prompts)

        # concatenate the long answer embeddings to the prompts and labels
        for i in range(batch_size):
            row = prompt_embeddings[i]
            if indices[i] == -1:
                pass
            else:
                # Replace the special token with the long answer embeddings
                row[indices[i]] = long_answer_embeddings[i]

        # obtain the loss from the decoder
        outputs = self.decoder.forward(input_embeds=prompt_embeddings, attention_mask=prompt_masks, labels=labels)
        return outputs.loss