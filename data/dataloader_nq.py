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

        for row in batch:
            # get the question from the row
            question = row["question"]["text"]
            # runs tokens through tokenizer and obtain numerical tokens
            # tokens, mask = self.tokenize(question, self.tokenizer)
            questions.append(question)
            # questions.append(tokens)
            # questions.append(tokens)

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
              # print(string_tokens)
              return string_tokens
            #   return self.tokenize(string_tokens, self.tokenizer)

            # obtain the sentence embedding
            answers.append(get_string_tokens(correct_index, candidates, doc))
            answers.append(get_string_tokens(wrong1, candidates, doc))
            answers.append(get_string_tokens(wrong2, candidates, doc))

            # check if true or false
            # tune the ratio during training
            labels.append(1)
            labels.append(0)
            labels.append(0)
        # TODO - tokenize all the questions and the answers at the end
        #

        # add padding to the questions and answers
        # questions = self.padding(questions)
        # print(f"questions={questions}")
        # print(f"answers={answers}")
        questions = self.tokenizer(questions, padding=True, max_length=int(max_len/4), truncation=True, return_tensors="pt")
        answers = self.tokenizer(answers, padding=True, max_length=int(max_len), truncation=True, return_tensors="pt")

        # requires the padding mask of answers
        # set a shorter max length to questions (128) as compared to answers (perhaps 512)
        # best practice to find out the ave length of questions and answers
        question_mask = questions["attention_mask"]
        answer_mask = answers["attention_mask"]
        questions = torch.LongTensor(questions["input_ids"])
        answers = torch.LongTensor(answers["input_ids"])

        # print(f"question size={questions.shape}")
        # print(f"qmask size={question_mask.shape}")
        # print(f"answer size={answers.shape}")
        # print(f"amask size={answer_mask.shape}")
        # TODO - add attention masks for the questions and the answers
        return {"questions": questions, "question_mask": question_mask, "long_answers": answers, "answer_mask": answer_mask, "labels": torch.LongTensor(labels)}

if __name__=="__main__":
    nq_dataloader = ModelDataLoader(batch_size=16)
    for batch in nq_dataloader:
        print(batch["questions"].shape)
        print(batch["long_answers"].shape)
        print(batch["labels"].shape)
        break