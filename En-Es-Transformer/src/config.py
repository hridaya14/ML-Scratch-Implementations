import torch

#Hyper Parameters
batch_size = 32
block_size = 100
random_seed = 42
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
eval_iters = 200
n_embd = 512
n_head = 4
n_layer = 6
dropout = 0.1
#--------------

tokenizer_config = {
    "src_lang" : "eng",
    "tgt_lang" : "es",
    'src_vocab_size': 10000,
    'tgt_vocab_size' : 10000
}

