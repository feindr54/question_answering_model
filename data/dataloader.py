import torch

from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from conf import question_max_len, long_answer_max_len, short_answer_max_len, prompt_max_len, device

class NQDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, mode="train"):
        if mode == "train":
            self.is_train = True
        elif mode == "valid":
            self.is_train = False
        else:
            raise ValueError(f"mode should be selected from train/valid, but got {mode}.")

        self.gpt_tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.gpt_tokenizer.add_special_tokens({'pad_token': '[PAD]', 'additional_special_tokens': ['[LA_SEP]']})

        self.bert_tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
        self.bert_tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        super(NQDataLoader, self).__init__(dataset, batch_size=batch_size, shuffle=False, drop_last=True, collate_fn=self.collate_fn)
    def collate_fn(self, batch):
        # obtain the required strings
        questions = []
        long_answers = []
        filler_tokens = self.gpt_tokenizer("[PAD][LA_SEP][PAD]Short answer: ", return_tensors='pt')
        short_answers_prompts = []
        prompts = []
        labels = []

        batch_size = len(batch)
        index = 0
        for row in batch:
            questions.append(row["question"])
            # correct long answer
            long_answers.append(row["correct_long_answer"])
            label_row = torch.zeros(size=(batch_size, ), dtype=torch.int)
            label_row[index] += 1
            labels.append(label_row)
            # wrong long answer
            for wrong_answer in row["wrong_long_answers"]:
                long_answers.append(wrong_answer)
                labels.append(torch.zeros(size=(batch_size, ), dtype=torch.int))

            short_answers_prompts.append(row["short_answer"])

            prompts.append(row["prompt"])

        # tokenize the questions
        q_output = self.bert_tokenizer(questions, padding=True, max_length=question_max_len, truncation=True, return_tensors="pt")
        q_tokens = q_output["input_ids"]
        q_mask = q_output["attention_mask"]

        # tokenize the long answers
        la_output = self.bert_tokenizer(long_answers, padding=True, max_length=long_answer_max_len, truncation=True, return_tensors="pt")
        la_tokens = la_output["input_ids"]
        la_mask = la_output["attention_mask"]

        # tensorize the long answer labels
        labels = torch.stack(labels, dim=0).to(torch.int)

        # tokenize the prompt
        prompt_output = self.gpt_tokenizer(prompts, padding=True, max_length=prompt_max_len, truncation=True, return_tensors="pt")
        prompt_tokens = prompt_output["input_ids"]
        prompt_mask = prompt_output["attention_mask"]

        # create the short answer labels

        sa_output = self.gpt_tokenizer(short_answers_prompts, padding=True, max_length=short_answer_max_len, truncation=True, return_tensors="pt")
        sa_tokens = sa_output["input_ids"]
        sa_mask = sa_output["attention_mask"]

        # generates the prompt tokens and prompt mask
        filler_tokens = filler_tokens["input_ids"].repeat(batch_size, 1)
        output_prompt_tokens, output_prompt_mask = self.generate_prompt_tokens(prompt_tokens, filler_tokens, sa_tokens, prompt_mask, sa_mask)

        # TODO - add a end token at the end of short answer tokens
        prompt_labels = torch.full_like(input=prompt_tokens, fill_value=-100)
        sa_labels = sa_output["input_ids"] * sa_output["attention_mask"] + (~sa_output["attention_mask"] * -100) # sets label of any element with attention mask 0 to -100
        prompt_labels = torch.cat((prompt_labels, filler_tokens, sa_labels), dim=1)

        # checks that the labels and the tokens have the same shape
        if (prompt_labels.shape != output_prompt_tokens.shape):
            print(f"tokens:{output_prompt_tokens.shape}")
            print(f"labels:{prompt_labels.shape}")

        return {"questions": q_tokens, "question_mask": q_mask, "long_answers": la_tokens, "long_answer_mask": la_mask,
                "long_answers_labels": labels, "prompts": output_prompt_tokens, "prompt_mask": output_prompt_mask, "short_answers_labels": prompt_labels}

    def generate_prompt_tokens(self, prompt_tokens, filler_tokens, short_answer_tokens, prompt_mask, short_answer_mask):
        # concat the prompt, filler word, and the short answer
        # filler_tokens = filler_tokens.repeat((len(prompt_tokens), 1))
        tokens = torch.cat((prompt_tokens, filler_tokens, torch.full_like(short_answer_tokens, fill_value=0)), dim=1)
        mask = torch.cat((prompt_mask, torch.full_like(input=filler_tokens, fill_value=1), torch.full_like(short_answer_mask, fill_value=0)), dim=1)
        return tokens, mask

if __name__ ==  "__main__":
    from .dataset import NQDataset
    ds = NQDataset()
    dl = NQDataLoader(ds, 2)
    for batch in dl:
        print(f"questions={batch['questions'].shape}")
        print(f"question_mask={batch['question_mask'].shape}")
        print(f"long_answers={batch['long_answers'].shape}")
        print(f"long_answers_mask={batch['long_answer_mask'].shape}")
        print(f"labels={batch['long_answers_labels'].shape}")
        print(f"prompts={batch['prompts'].shape}")
        print(f"prompt_mask={batch['prompt_mask'].shape}")
        # print(f"prompt_labels={batch['prompt_labels'].shape}")
        print(f"short_answers_labels={batch['short_answers_labels'].shape}")
        break
