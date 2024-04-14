import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import Adam
import numpy as np
from transformers import BertModel
import time

from conf import *
from util.epoch_timer import epoch_time
from data.dataloader_nq import ModelDataLoader as Dataloader

# class CrossAttentionLayer(nn.Module):
#     def __init__(self, device):
#         super(CrossAttentionLayer, self).__init__()
#         self.device = device
#     def forward(self, query, key, value):
#         query = query.to(device) # batch * a * 768
#         key = key.to(device) # batch * q * 768
#         out = torch.softmax(torch.bmm(query, key.T), dim=1) # batch * a * q
#         out = torch.bmm(out, value) # must multiply by batch * q * encoding_length
#         return out

class QAModel(nn.Module):
    def __init__(self, device):
        super(QAModel, self).__init__()
        self.qbert = BertModel.from_pretrained("google-bert/bert-base-uncased").to(device)
        self.abert = BertModel.from_pretrained("google-bert/bert-base-uncased").to(device)

        # self.cx_attention = [CrossAttentionLayer(device) for _ in range(6)]
        self.cx_attention = [nn.MultiheadAttention(768, 2, batch_first=True, device=device) for _ in range(6)] # output remains query size, ie abatch x seqlen x 768
        self.linear = nn.Linear(768,1, device=device) # input is batch * answers * 768

        self.device = device
    def forward(self, question, answer, question_mask, answer_mask):
        # obtain bert embeddings of question and answer with attention mask
        # get the pooled output for the questions, and the unpooled output from the answers
        q_embeds = self.qbert(question, question_mask).pooler_output.to(self.device) # (abatch) * qbatch x 768
        a_embeds = self.abert(answer, answer_mask).last_hidden_state.to(self.device) # abatch x seqlen x 768
        # q_embeds = q_embeds.unsqueeze(0)
        q_embeds = q_embeds.repeat(a_embeds.shape[0],1,1)

        print(f"q_embed shape:{q_embeds.shape}")
        print(f"a_embed shape:{a_embeds.shape}")

        query = q_embeds
        key = a_embeds
        value = a_embeds
        # run query through cross attention
        for cx_layer in self.cx_attention:
            query = cx_layer(query, key, value)[0] # 0th index is attn output, 1st index is attention weights

        # TODO - linear layer
        print(f"cx_attention output shape:{query.shape}")
        output = self.linear(query)
        print(f'output_shape:{output.shape}')
        # output = self.sigmoid(self.linear(value))
        return output

model = QAModel(device)

train_dataloader = Dataloader(batch_size=batch_size, mode="train")
val_dataloader = Dataloader(batch_size=batch_size, mode="valid")

optimizer = nn.Adam(params=model.parameters(),
                 lr=init_lr,
                 weight_decay=weight_decay,
                 eps=adam_eps)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                 verbose=True,
                                                 factor=factor,
                                                 patience=patience)
criterion = nn.BCEWithLogitsLoss()

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
        labels = batch['labels'].to(device)
        optimizer.zero_grad()
        # run Bert on both
        y = model(questions, answers, question_mask, answer_mask)
        loss = criterion(y, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        data_num += labels.questions[0]
        print('step :', i, ', loss :', loss.item())
    return epoch_loss / data_num

def evaluate(model, criterion):
    model.eval()
    epoch_loss = 0
    data_num = 0
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

if __name__ == "__main__":
    run(total_epoch=epoch, best_loss=inf)