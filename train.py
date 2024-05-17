import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import Adam
import numpy as np
from transformers import BertModel, GPT2LMHeadModel, AutoTokenizer
import time

from conf import *
from data.dataloader_nq import ModelDataLoader as Dataloader
from data.dataloader_nq import DecoderDataLoader

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

"""
Args
- model: pre-trained decoder model
- batch: each includes questions, correct short answers and correct long answer
"""
def train_decode(model, batch):
    pass

class QAShortAnswer(nn.Module):
    def __init__(self, device):
        super(QAShortAnswer, self).__init__(device=device)
        self.decoder = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")

        # need a long answer bank, with the appropriate embeddings
        # TODO - load a long answer bank (pre-trained long answer retriever)

    def forward(self, tokens):
        # TODO - design a prompt for the decoder
        question = ...
        # TODO - create a template for documents
        documents = ...
        predict_prompt = "Question: " + question + "Long answer: " + documents + "Long answer embeddings: " + ... + "Short Answer: "

        # pass the tokens into the decoder
        # if training, then just call the forward function and generate a loss

    def create_document_text(long_answers):
        pass

class LongAnswerModel(nn.Module):
    def __init__(self, device):
        super(LongAnswerModel, self).__init__()
        self.qbert = BertModel.from_pretrained("google-bert/bert-base-uncased").to(device)
        self.abert = BertModel.from_pretrained("google-bert/bert-base-uncased").to(device)

        self.cx_attention = [] # output remains query size, ie abatch x seqlen x 768
        self.attention_layers = 6

        for _ in range(6):
            self.cx_attention.append(nn.MultiheadAttention(768, 2, batch_first=True, device=device))
            self.cx_attention.append(nn.Linear(in_features=768, out_features=768, device=device))
            self.cx_attention.append(nn.ReLU())
        # add some feedforward and ReLU (input output same shape) between cross attention
        self.linear = nn.Linear(768,1, device=device) # input is batch * answers * 768
        self.device = device

    def forward(self, question, answer, question_mask, answer_mask):
        # obtain bert embeddings of question and answer with attention mask
        # get the pooled output for the questions, and the unpooled output from the answers
        q_embeds = self.qbert(question, question_mask).pooler_output.to(self.device) # (abatch) * qbatch x 768
        a_embeds = self.abert(answer, answer_mask).last_hidden_state.to(self.device) # abatch x seqlen x 768

        # extend the dimensions of the questions to the answer batch
        q_embeds = q_embeds.repeat(a_embeds.shape[0],1,1)

        # print(f"q_embed shape:{q_embeds.shape}")
        # print(f"a_embed shape:{a_embeds.shape}")

        query = q_embeds
        key = a_embeds
        value = a_embeds
        # run query through cross attention
        for i in range(self.attention_layers):
            # cross attention layer
            query = self.cx_attention[i*3](query, key, value)[0] # 0th index is attn output, 1st index is attention weights
            # feedforward layer
            query = self.cx_attention[i*3+2](self.cx_attention[i*3+1](query))


        # final linear layer to convert to binary value
        output = self.linear(query)
        return output.squeeze(-1)

class ShortAnswerModel(device):
    """
    Constructor of the Short Answer Model
    """
    def __init__(self, device):
        super(ShortAnswerModel, self).__init__(device=device)
        self.device = device
        self.decoder = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")

    """
    Forward pass of the short answer model.

    Args:
    - long_answer_ids: long answer input ids (tokens)
    - long_answer_mask: long answer attention masks

    """
    def forward(self, long_answer_ids, long_answer_mask, prompts, prompt_masks, labels):
        # obtain the indices of special sep token for each prompt (1D tensor)
        # id of special [LA_SEP] token = 30522
        # TODO - find a way to not hardcode it
        special_token = 30522
        matches = (prompts == special_token)
        indices = torch.argmax(matches.int(), dim=1)
        # sets any non-matches to -1
        indices[~matches.any(dim=1)] = -1

        # convert the input ids to input embeddings
        with torch.no_grad():
            prompt_embeddings = self.decoder.wte(prompts)
            la_embeddings = self.decoder.wte(long_answer_ids)

        # concatenate the long answer embeddings to the prompts and labels
        embeddings = []
        masks = []
        for i in range(batch_size):
            row = prompt_embeddings[i]
            if indices[i] == -1:
                pass
            else:
                first_half = prompt_embeddings[i, :indices[i]]
                second_half = prompt_embeddings[i, indices[i+1:]]
                embeddings.append(torch.cat(first_half, la_embeddings[i], second_half))

                left_mask = prompt_masks[i, :indices[i]]
                right_mask = prompt_masks[i, indices[i+1:]]
                masks.append(torch.cat(left_mask, long_answer_mask[i], right_mask))
        embeddings = torch.stack(embeddings)
        masks = torch.stack(masks)

        # obtain the loss from the decoder
        outputs = self.decoder.forward(input_embeds=embeddings, attention_mask=masks, labels=labels)
        return outputs.loss

class QAModel(device):
    """
    Constructor of the QA model
    """
    def __init__(self, device):
        super(QAModel, self).__init__()
        # contains a long answer model and short answer model
        self.long_answer_model = LongAnswerModel(device)
        self.short_answer_model = ShortAnswerModel(device)
        self.device = device

    """
    Determines the forward function of the overall QA model

    Args:
    - la_inputs: long answer inputs, Tuple of
        - question tokens: 2D tensor
        - answer tokens: 2D tensor
        - question mask: 2D tensor
        - answer mask: 2D tensor
    - sa_inputs: short answer inputs, Tuple of (long answer ids obtained from la_inputs)
        - prompts: 2D tensor of tokens
        - prompt_mask: 2D tensor of tokens
        - labels: 2D tensor of tokens

    Returns:
    - losses: Tuple of
        - la_loss: resulting from the long answer model (encoder)
        - sa_loss: resulting from the short answer model (decoder)
    """
    def forward(self, la_inputs, sa_inputs):
        # run the forward pass to the long answer retriever, and retrieve a long answer embedding
        qtokens, atokens, qmask, amask = la_inputs
        la_loss = self.long_answer_model(qtokens, atokens, qmask, amask)

        # run the forward pass of the short answer model
        prompts, prompt_mask, labels = sa_inputs
        sa_loss = self.short_answer_model(atokens, amask, prompts, prompt_mask, labels)

        return la_loss, sa_loss

    # TODO - function that generates a short answer from a given input
    def generate(self):
        pass

model = QAModel(device)

train_dataloader = Dataloader(batch_size=batch_size, mode="train")
# val_dataloader = Dataloader(batch_size=batch_size, mode="valid")

optimizer = Adam(params=model.parameters(),
                 lr=init_lr,
                 weight_decay=weight_decay,
                 eps=adam_eps)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                 verbose=True,
                                                 factor=factor,
                                                 patience=patience)
criterion = nn.BCEWithLogitsLoss()

def decoder_train(optimizer):
    epoch_loss = 0
    data_num = 0

    model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")
    model.train()
    dataloader = DecoderDataLoader(device)
    for i, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        mask = batch["mask"].to(device)
        labels = batch["labels"].to(device)
        optimizer.zero_grad()

        # run the decoder on the input
        loss = model(input_ids, mask, labels=labels)
        loss.backward()
        optimizer.step()

        # update total loss
        epoch_loss += loss.item()
        data_num += len(labels)
    return epoch_loss / data_num


def train(model, optimizer, criterion):
    # set model to training mode
    model.train()
    epoch_loss = 0
    data_num = 0
    for i, batch in enumerate(train_dataloader):
        questions = batch['questions'].to(device)
        question_mask = batch['question_mask'].to(device)
        answers = batch['long_answers'].to(device)
        answer_mask = batch['answer_mask'].to(device)
        labels = batch['labels'].float().to(device)
        optimizer.zero_grad()
        # run Bert on both
        y = model(questions, answers, question_mask, answer_mask)
        loss = criterion(y, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        data_num += len(labels)
        print('step :', i, ', loss :', loss.item())
    return epoch_loss / data_num

def evaluate(model, criterion):
    model.eval()
    epoch_loss = 0
    data_num = 0
    val_dataloader = Dataloader(batch_size=batch_size, mode="valid")
    with torch.no_grad():
        for i, batch in enumerate(val_dataloader):
            # obtain the embedding for the question and answers
            questions = batch['questions'].to(device)
            question_mask = batch['question_mask'].to(device)
            answers = batch['long_answers'].to(device)
            answer_mask = batch['answer_mask'].to(device)
            labels = batch['labels'].to(device)
            # run the model on the question and answers
            y = model(questions, answers, question_mask, answer_mask)
            loss = criterion(y, labels)

            epoch_loss += loss.item()
            data_num += questions.shape[0]
    return epoch_loss / data_num

def run(total_epoch, best_loss):
    import math
    train_losses, test_losses = [], []
    for step in range(total_epoch):
        # train and evaluate model
        start_time = time.time()
        train_loss = train(model, optimizer, criterion)
        valid_loss = evaluate(model, criterion)
        end_time = time.time()

        train_losses.append(train_loss)
        test_losses.append(valid_loss)

        # time the process
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        # saves the model with the best loss
        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), 'saved/model-{0}.pt'.format(valid_loss))

        f = open('result/train_loss.txt', 'w')
        f.write(str(train_losses))
        f.close()

        f = open('result/test_loss.txt', 'w')
        f.write(str(test_losses))
        f.close()

        print(f'Epoch: {step + 1} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\tVal Loss: {valid_loss:.3f} |  Val PPL: {math.exp(valid_loss):7.3f}')

def decoder_run(total_epoch, best_loss):
    import math
    train_losses, test_losses = [], []
    for step in range(total_epoch):
        # train and evaluate model
        start_time = time.time()
        train_loss = decoder_train(optimizer, criterion)
        # valid_loss = evaluate(model, criterion)
        end_time = time.time()

        train_losses.append(train_loss)
        # test_losses.append(valid_loss)

        # time the process
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        # saves the model with the best loss
        # if valid_loss < best_loss:
        #     best_loss = valid_loss
        #     torch.save(model.state_dict(), 'saved/model-{0}.pt'.format(valid_loss))
        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(model.state_dict(), 'saved/model-{0}.pt'.format(train_loss))

        f = open('result/train_loss.txt', 'w')
        f.write(str(train_losses))
        f.close()

        # f = open('result/test_loss.txt', 'w')
        # f.write(str(test_losses))
        # f.close()

        print(f'Epoch: {step + 1} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        # print(f'\tVal Loss: {valid_loss:.3f} |  Val PPL: {math.exp(valid_loss):7.3f}')
if __name__ == "__main__":
    run(total_epoch=epoch, best_loss=inf)