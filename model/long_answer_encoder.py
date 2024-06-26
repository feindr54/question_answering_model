import torch.nn as nn

from transformers import BertModel

class LongAnswerEncoder(nn.Module):
    def __init__(self, device) -> None:
        super(LongAnswerEncoder, self).__init__()
        self.abert = BertModel.from_pretrained("google-bert/bert-base-uncased").cuda(device)

    def forward(self, long_answer, long_answer_mask):
        a_embeds = self.abert(long_answer, long_answer_mask).last_hidden_state#.to(device=self.device) # (abatch) * qbatch x 768
        return a_embeds

if __name__ == "__main__":
    import torch
    from conf import device
    encoder = LongAnswerEncoder(device=device)
    input_tokens = torch.randint(size=(30,512), low=0, high=20000).cuda()
    answer_mask = torch.full_like(input_tokens, fill_value=1)
    output = encoder(input_tokens, answer_mask)
    print(f"output shape={output.shape}")
