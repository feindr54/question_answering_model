import torch
from torch.utils.data import DataLoader
from datasets import load_dataset

from transformers import AutoTokenizer
import numpy as np

# from conf import max_len
max_len = 512

def get_dataset():
  data = load_dataset("natural_questions")
  train = data['train']
  validation = data['validation']
  return train, validation

class DecoderDataLoader(DataLoader):
    def __init__(self, batch_size, mode="train"):
        train, validation = get_dataset()
        if mode == "train":
            data = train
            self.is_train = True
        elif mode == "valid":
            data = validation
            self.is_train = False
        else:
            raise ValueError(f"mode should be selected from train/valid, but got {mode}.")

        super(ModelDataLoader, self).__init__(data, batch_size=batch_size, shuffle=False, drop_last=True, collate_fn=self.collate_fn)

    def collate_fn(self, batch):
        # create a prompt based on the correct answer
        prompts = []
        valid_prompts = []
        for row in batch:
            # get the question from the row
            question = row["question"]["text"]
            # add questions to a list of questions
            # questions.append(question)

            # attach the short answers together
            annotations = row["annotations"]["short_answers"] # list of short answers
            s_a_prompt: str = ""
            for short_answer in annotations:
                s_a_prompt += short_answer["text"] + ","
            s_a_prompt = s_a_prompt[:-1]

            #obtain a long answer candidate
            doc = row['document']
            long_answer = row["annotations"]["long_answer"][0]
            correct_index = long_answer["candidate_index"]
            candidates = row['long_answer_candidates']

            def get_string_tokens(index, candidates, doc):
              start_token = candidates["start_token"][index]
              end_token = candidates["end_token"][index]

              # remove the html tags (irrelevant text)
              # can consider keeping the html
              is_not_html = np.invert(np.array(doc["tokens"]["is_html"]))[start_token:end_token]
              tokens = np.array(doc["tokens"]["token"])[start_token:end_token]
              string_tokens = " ".join(list(tokens[is_not_html]))

              return string_tokens
            l_a_prompt:str = get_string_tokens(correct_index, candidates, doc)

            # generates the prompt; if training set, append actual short answer to the end
            prompt = "Question: " + question + " Answers: " + l_a_prompt + ". Short answer: "
            valid_prompts.append(prompt)
            if (self.is_train):
                prompt += s_a_prompt
            prompts.append(prompt)
        # convert the list of prompts into a list of input ids, and corresponding labels
        # encode all the strings, with padding, and get the required attention mask for each
        train_prompt_tokens = self.tokenizer(prompts, padding=True, max_length=256, truncation=True, return_tensors="pt")
        input_ids = train_prompt_tokens["input_ids"]
        valid_prompt_tokens = self.tokenizer(valid_prompts, padding=True, max_length=256, truncation=True, return_tensors="pt")
        # perform bit-wise and on both attention masks for the final mask
        attention_mask = train_prompt_tokens["attention_mask"] * valid_prompt_tokens["attention_mask"]
        # generate the labels from the new mask
        labels = train_prompt_tokens * attention_mask # all entries all either original token values or 0
        # convert the 0s to -100
        labels = torch.where(input_ids > 0, input_ids, -100)
        return {"input_ids": input_ids, "attention_mask": train_prompt_tokens["attention_mask"], "labels": labels}

class ModelDataLoader(DataLoader):
    def __init__(self, batch_size, mode="train"):
        # get the data
        train, validation = get_dataset()
        if mode == "train":
            data = train
        elif mode == "valid":
            data = validation
        else:
            raise ValueError(f"mode should be selected from train/valid, but got {mode}.")

        self.tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
        self.pad_idx = 0


        # TODO: sort by the questions/answers

        # want: question(tokenized); candidate long answer (tokenized); true or false entries
        # for each question, get the correct answer, and get 3 irrelevant answer
        super(ModelDataLoader, self).__init__(data, batch_size=batch_size, shuffle=False, drop_last=True, collate_fn=self.collate_fn)

    def tokenize(self, text, tokenizer):
        # TODO - can specify max length, padding, truncating, with correct impl don't need padding function
        # TODO - return
        tokenized_result = tokenizer(text)
        tokens = tokenized_result["input_ids"]
        mask = tokenized_result["attention_mask"]
        return tokens, mask

    def padding(self, tokens, max_len):
        seq_len = min(max([len(s) for s in tokens]), max_len)
        return [(s + [self.pad_idx] * (seq_len - len(s)) if len(s) < seq_len else s[:seq_len] ) for s in tokens]

    def collate_fn(self, batch):
        questions = []
        answers = []
        labels = []

        index = 0
        q_count = len(batch)
        for row in batch:
            # get the question from the row
            question = row["question"]["text"]
            # add questions to a list of questions
            questions.append(question)

            # get a correct answer and 2 wrong answers in the same batch
            doc = row['document']
            annotations = row["annotations"]["long_answer"][0]
            correct_index = annotations["candidate_index"]
            candidates = row['long_answer_candidates']

            # get 2 random and wrong indices
            wrong1 = correct_index
            wrong2 = correct_index
            while (wrong1 == correct_index or wrong2 == correct_index):
                wrong1 = np.random.randint(0, len(candidates["top_level"]))
                wrong2 = np.random.randint(0, len(candidates["top_level"]))

            def get_string_tokens(index, candidates, doc):
              start_token = candidates["start_token"][index]
              end_token = candidates["end_token"][index]

              # remove the html tags (irrelevant text)
              # can consider keeping the html
              is_not_html = np.invert(np.array(doc["tokens"]["is_html"]))[start_token:end_token]
              tokens = np.array(doc["tokens"]["token"])[start_token:end_token]
              string_tokens = " ".join(list(tokens[is_not_html]))

              return string_tokens

            # obtain the sentence embedding
            answers.append(get_string_tokens(correct_index, candidates, doc))
            answers.append(get_string_tokens(wrong1, candidates, doc))
            answers.append(get_string_tokens(wrong2, candidates, doc))

            # check if true or false
            # tune the ratio during training
            zeros = torch.zeros(size=(q_count,), dtype=torch.long)
            zeros[index] += 1
            labels.append(zeros)
            labels.append(torch.zeros(size=(q_count, ), dtype=torch.long))
            labels.append(torch.zeros(size=(q_count, ), dtype=torch.long))
            index += 1

        # tokenize and add padding to the questions and answers
        questions = self.tokenizer(questions, padding=True, max_length=int(max_len/4), truncation=True, return_tensors="pt")
        answers = self.tokenizer(answers, padding=True, max_length=int(max_len), truncation=True, return_tensors="pt")

        # requires the padding mask of answers
        # set a shorter max length to questions (128) as compared to answers (perhaps 512)
        # best practice to find out the ave length of questions and answers
        question_mask = questions["attention_mask"]
        answer_mask = answers["attention_mask"]
        questions = torch.LongTensor(questions["input_ids"])
        answers = torch.LongTensor(answers["input_ids"])

        # create a label 2d tensor from a list of tensors
        labels = torch.stack(labels, dim=0) # answer x question, 0, 3, 6,

        # add attention masks for the questions and the answers
        return {"questions": questions, "question_mask": question_mask, "long_answers": answers, "answer_mask": answer_mask, "labels": labels}

if __name__=="__main__":
    nq_dataloader = ModelDataLoader(batch_size=16)
    for batch in nq_dataloader:
        print(batch["questions"].shape)
        print(batch["long_answers"].shape)
        print(batch["labels"].shape)
        break