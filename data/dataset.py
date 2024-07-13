import numpy as np

from datasets import load_from_disk
from torch.utils.data import Dataset, IterableDataset
from utils.get_dataset import get_dataset

class NQIterDataset(IterableDataset):
    def __init__(self) -> None:
        super().__init__()
        self.train, self.valid = get_dataset()

    def __iter__(self):
        for row in self.train:
            # if there is no long answers, skip
            # if there are no short answers, empty string short answers, then yield
            yield row
"""
Dataset contains a list of questions, with corresponding long answers (and the correct one), and the corresponding
short answer (should I add a label too?)
"""
class NQDataset(Dataset):
    def __init__(self, train=True):
        super(NQDataset, self).__init__()
        self.is_train = train
        self.train, self.valid = get_dataset()
        self.data = self.train if train else self.valid
        self.len = len(self.train) if train else len(self.valid)
        # print(len(indices), ", ", len(self.data))
    def _preprocess(self, data):
        indices = []
        index = 0
        for row in data:
            # check if no correct long answer
            annotations = row["annotations"]["long_answer"][0]
            print("number of long answer candidates: ", len(row['long_answer_candidates']['top_level']))
            print(row["annotations"])
            correct_index = annotations["candidate_index"]
            if correct_index == -1:
                # no short answer
                index += 1
                continue
            if len(row["annotations"]["short_answers"][0]["text"]) == 0 and row["annotations"]["yes_no_answer"] == -1:
                index += 1
                continue
            else:
                indices.append(index)
                index += 1
        return indices

    def save_preprocessed_dataset(self):
        import time
        start = time.time()
        train_indices = self._preprocess(self.train)
        end = time.time()
        print(end - start , " seconds")
        train = self.train.select(train_indices)
        # valid_indices = self._preprocess(self.valid)
        # valid = self.valid.select(valid_indices)
        train.save_to_disk("/scratch/gilbreth/gao654/rag/train")
        # valid.save_to_disk("/scratch/gilbreth/gao654/rag/valid")

    def load_dataset(self):
        if self.is_train:
            self.data = load_from_disk("/scratch/gilbreth/gao654/rag/train")
        else:
            self.data = load_from_disk("/scratch/gilbreth/gao654/rag/valid")

    def get_string_tokens(self, index, candidates, doc):
        start_token = candidates["start_token"][index]
        end_token = candidates["end_token"][index]

        # remove the html tags (irrelevant text)
        is_not_html = np.invert(np.array(doc["tokens"]["is_html"]))[start_token:end_token]
        tokens = np.array(doc["tokens"]["token"])[start_token:end_token]
        string_tokens = " ".join(list(tokens[is_not_html]))

        return string_tokens

    def __len__(self):
        return self.len
    def __getitem__(self, index):
        row = self.data[index]

        # obtain the question
        question = row["question"]["text"]

        # get a correct answer and 2 wrong answers in the same batch
        doc = row['document']
        annotations = row["annotations"]["long_answer"][0]
        correct_index = annotations["candidate_index"]
        candidates = row['long_answer_candidates']

        print("correct index: ", correct_index)
        print("number of long answer candidates: ", len(row['long_answer_candidates']['start_token']))

        # check whether there are long answer candidates
        if len(row['long_answer_candidates']['start_token']) == 0:
            print("no long answer candidates")
            return {"question": question,
                    "correct_long_answer": None,
                    "wrong_long_answers": None,
                    "short_answer": None,
                    "prompt": None}

        # TODO - check whether there is a long answer
        if (len(row['long_answer_candidates']['start_token']) == 0):
            wrong_answers = []
            if len(row['long_answer_candidates']['start_token']) == 1:
                wrong1 = 0
                wrong_answer1 = self.get_string_tokens(wrong1, candidates, doc)
                wrong_answers.append(wrong_answer1)
            elif len(row['long_answer_candidates']['start_token']) == 2:
                wrong1 = 0
                wrong2 = 1
                wrong_answer1 = self.get_string_tokens(wrong1, candidates, doc)
                wrong_answer2 = self.get_string_tokens(wrong2, candidates, doc)
                wrong_answers.append(wrong_answer1)
                wrong_answers.append(wrong_answer2)
            elif len(row['long_answer_candidates']['start_token']) == 3:
                wrong1 = 0
                wrong2 = 1
                wrong3 = 2
                wrong_answer1 = self.get_string_tokens(wrong1, candidates, doc)
                wrong_answer2 = self.get_string_tokens(wrong2, candidates, doc)
                wrong_answer3 = self.get_string_tokens(wrong3, candidates, doc)
                wrong_answers.append(wrong_answer1)
                wrong_answers.append(wrong_answer2)
                wrong_answers.append(wrong_answer3)
            else:
            # no correct long answer
                wrong1 = np.random.randint(0, len(candidates['start_token']))
                wrong2 = np.random.randint(0, len(candidates['start_token']))
                wrong3 = np.random.randint(0, len(candidates['start_token']))

                wrong_answer1 = self.get_string_tokens(wrong1, candidates, doc)
                wrong_answer2 = self.get_string_tokens(wrong2, candidates, doc)
                wrong_answer3 = self.get_string_tokens(wrong3, candidates, doc)

                wrong_answers.append(wrong_answer1)
                wrong_answers.append(wrong_answer2)
                wrong_answers.append(wrong_answer3)

            s_a_prompt = "<|endoftext|>"

            # generate the prompt
            prompt = "Question: " + question + " [PAD]Answers: " + correct_long_answer + "[PAD][LA_SEP][PAD]Short answer: " # + tokenier_pad_token + short answer + right paddings

            return {"question": question,
                    "correct_long_answer": None,
                    "wrong_long_answers": wrong_answers,
                    "short_answer": s_a_prompt,
                    "prompt": prompt}
        else:
            correct_long_answer = self.get_string_tokens(correct_index, candidates, doc)

            # extract long answers, and the correct long answers
            # get 2 random and wrong indices, there are at least 3 candidates
            wrong_answers = []
            if len(candidates['start_token']) == 1:
                pass
            elif len(candidates['start_token']) == 2:
                if (correct_index == 0):
                    wrong1 = 1
                else:
                    wrong1 = 0

                wrong_answer1 = self.get_string_tokens(wrong1, candidates, doc)
                wrong_answers.append(wrong_answer1)
            else:
                wrong1 = correct_index
                wrong2 = correct_index
                while (wrong1 == correct_index or wrong2 == correct_index):
                    wrong1 = np.random.randint(0, len(candidates['start_token']))
                    wrong2 = np.random.randint(0, len(candidates['start_token']))

                wrong_answer1 = self.get_string_tokens(wrong1, candidates, doc)
                wrong_answer2 = self.get_string_tokens(wrong2, candidates, doc)
                wrong_answers.append(wrong_answer1)
                wrong_answers.append(wrong_answer2)

            print("short answers: ", row["annotations"]["short_answers"][0]["text"])
            print("yes no answer: ", row["annotations"]["yes_no_answer"])
            # check if there is a short answer
            if len(row["annotations"]["short_answers"][0]["text"]) == 0:
                if (row["annotations"]["yes_no_answer"][0] == 1): # yes
                    s_a_prompt = "yes"
                # no short answer
                elif (row["annotations"]["yes_no_answer"][0] == 0): #no
                    s_a_prompt = "no"
                else:
                    s_a_prompt = ""
            else:
                # extract short answers
                short_answer_annotations = row["annotations"]["short_answers"] # list of short answers
                s_a_prompt: str = ""
                for short_answer in short_answer_annotations:
                    print(short_answer["text"])
                    s_a_prompt += short_answer["text"][0] + ","
                s_a_prompt = s_a_prompt[:-1] # remove that last comma
                s_a_prompt += "<|endoftext|>" # add the end of sentence token

            # generate the prompt
            prompt = "Question: " + question + " [PAD]Answers: " + correct_long_answer + "[PAD][LA_SEP][PAD]Short answer: " # + tokenier_pad_token + short answer + right paddings

            return {"question": question,
                    "correct_long_answer": correct_long_answer,
                    "wrong_long_answers": wrong_answers,
                    "short_answer": s_a_prompt,
                    "prompt": prompt}

if __name__ == "__main__":
    # create a dataset and check the entries
    dataset = NQDataset()
    # dataset.save_preprocessed_dataset()
    for i in range(100):
        print("index ", i, ": ")
        print(dataset[i])
    # # print(dataset)
    # print(dataset[1])