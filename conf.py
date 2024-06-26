import torch

# GPU device setting
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# model parameter setting
batch_size = 8
max_len = 64
d_model = 512
n_layers = 6
n_heads = 8
ffn_hidden = 2048
drop_prob = 0.1

# optimizer parameter setting
init_lr = 5e-5
factor = 0.9
adam_eps = 5e-9
patience = 10
warmup = 100
epoch = 400
clip = 1.0
weight_decay = 5e-4
inf = float('inf')

# embedding max lengths
question_max_len = 24
long_answer_max_len = 512
prompt_max_len = 512
short_answer_max_len = 50
