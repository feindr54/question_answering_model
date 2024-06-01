import numpy as np

from torch.utils.data import Dataset
from utils.get_dataset import get_dataset

"""
Dataset contains a list of questions, with corresponding long answers (and the correct one), and the corresponding
short answer (should I add a label too?)
"""
class NQDataset(Dataset):
    def __init__(self):
        self.data, _ = get_dataset()
        self.len = len(self.data)

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
        correct_long_answer = self.get_string_tokens(correct_index, candidates, doc)

        # extract long answers, and the correct long answers
        # get 2 random and wrong indices
        wrong1 = correct_index
        wrong2 = correct_index
        while (wrong1 == correct_index or wrong2 == correct_index):
            wrong1 = np.random.randint(0, len(candidates))
            wrong2 = np.random.randint(0, len(candidates))

        wrong_answer1 = self.get_string_tokens(wrong1, candidates, doc)
        wrong_answer2 = self.get_string_tokens(wrong2, candidates, doc)

        # extract short answers
        short_answer_annotations = row["annotations"]["short_answers"] # list of short answers

        print(short_answer_annotations)

        s_a_prompt: str = ""
        for short_answer in short_answer_annotations:
            s_a_prompt += short_answer["text"][0] + ","
        s_a_prompt = s_a_prompt[:-1] # remove that last comma
        s_a_prompt += "<|endoftext|>" # add the end of sentence token

        # generate the prompt
        prompt = "Question: " + question + " [PAD]Answers: " + correct_long_answer + "[PAD][LA_SEP][PAD]Short answer: " # + tokenier_pad_token + short answer + right paddings

        return {"question": question,
                "correct_long_answer": correct_long_answer,
                "wrong_long_answers": [wrong_answer1, wrong_answer2],
                "short_answer": s_a_prompt,
                "prompt": prompt}

if __name__ == "__main__":
    # create a dataset and check the entries
    dataset = NQDataset()
    print(dataset)
    print(dataset[1])