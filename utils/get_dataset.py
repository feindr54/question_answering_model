from datasets import load_dataset

def get_dataset():
    data = load_dataset("natural_questions")
    train = data['train']
    validation = data['validation']
    return train, validation