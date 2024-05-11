import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import Adam
import numpy as np
from transformers import BertModel, GPT2LMHeadModel, AutoTokenizer
import time

from conf import *
from util.epoch_timer import epoch_time
from data.dataloader_nq import ModelDataLoader as Dataloader
from data.dataloader_nq import DecoderDataLoader

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
    def forward(self, tokens):
        # TODO - design a prompt for the decoder
        question = ...
        # TODO - create a template for documents
        documents = ...
        predict_prompt = question + "[PAD]" + documents + "[PAD]" + "Short Answer: "

        # pass the tokens into the decoder
        # if training, then just call the forward function and generate a loss

    def create_document_text(long_answers):
        pass

class QAModel(nn.Module):
    def __init__(self, device):
        super(QAModel, self).__init__()
        self.qbert = BertModel.from_pretrained("google-bert/bert-base-uncased").to(device)
        self.abert = BertModel.from_pretrained("google-bert/bert-base-uncased").to(device)

        self.cx_attention = [] # output remains query size, ie abatch x seqlen x 768
        for _ in range(6):
            self.cx_attention.append(nn.MultiheadAttention(768, 2, batch_first=True, device=device))
            self.cx_attention.append(nn.Linear(in_features=768, out_features=768))
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
        for i in range(6):
            # cross attention layer
            query = self.cx_attention[i](query, key, value)[0] # 0th index is attn output, 1st index is attention weights
            # feedforward layer
            query = self.cx_attention[i+2](self.cx_attention[i+1](query))


        # TODO - linear layer
        output = self.linear(query)
        return output.squeeze(-1)

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

def decoder_train(pptimizer):
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
        labels = batch['labels'].to(device).float()
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