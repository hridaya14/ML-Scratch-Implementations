#Dependencies
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import config

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
VOCAB_SIZE = config.tokenizer_config['src_vocab_size']
#--------------

class MultiHeadAttention(nn.Module):
    ''' Multihead attention for parallel computing'''
    def __init__(self, n_embd, n_head):
        super(MultiHeadAttention, self).__init__()
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"
        self.n_embd = n_embd
        self.n_head = n_head
        self.d_k = n_embd // n_head

        self.W_q = nn.Linear(n_embd, n_embd)
        self.W_k = nn.Linear(n_embd, n_embd)
        self.W_v = nn.Linear(n_embd, n_embd)
        self.W_o = nn.Linear(n_embd, n_embd)

    def scaled_attention(self, Q, K, V, mask=None):
        wei = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            wei = wei.masked_fill(mask == 0, -1e9)

        wei = torch.softmax(wei, dim=-1)
        output = torch.matmul(wei, V)
        return output

    def split_heads(self, x):
        batch_size, seq_len, n_embd = x.size()
        return x.reshape(batch_size, seq_len, self.n_head, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        batch_size, _, seq_len, d_k = x.size()
        return x.transpose(1, 2).reshape(batch_size, seq_len, self.n_embd)

    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        attn_output = self.scaled_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output


class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class PositionalEncoding(nn.Module):
    ''' Sine and Cos based positional encoding'''
    def __init__(self, n_embd, block_size):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(block_size, n_embd, device=device)
        position = torch.arange(0, block_size, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.pow(10_000, (-torch.arange(0, n_embd, 2, device=device).float() / n_embd))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return self.register_buffer('pe', pe.unsqueeze(0))


    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class EncoderBlock(nn.Module):
    ''' Encoder Block consisting of multiheaded attention and scaling'''
    def __init__(self, n_embd, n_head, dropout):
        super(EncoderBlock, self).__init__()

        self.self_attn = MultiHeadAttention(n_embd, n_head)
        self.feed_forward = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = self.ln1(x + self.dropout(self.self_attn(x, x, x, mask)))
        x = self.ln2(x + self.feed_forward(x))
        return x

class DecoderBlock(nn.Module):
    ''' Decoder block with Masked Self Attention and Crossover attention from encoder'''
    def __init__(self, n_embd, n_head, dropout):
        super(DecoderBlock, self).__init__()

        self.self_attn = MultiHeadAttention(n_embd, n_head)
        self.cross_attn = MultiHeadAttention(n_embd, n_head)
        self.feed_forward = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ln3 = nn.LayerNorm(n_embd)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x, enc_output, src_mask, tgt_mask):
        x = self.ln1(x + self.dropout(self.self_attn(x, x, x, tgt_mask)))
        x = self.ln2(x + self.dropout(self.cross_attn(x, enc_output, enc_output, src_mask)))
        x = self.ln3(x + self.feed_forward(x))
        return x


class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.encoder_embedding = nn.Embedding(VOCAB_SIZE + 1, n_embd , padding_idx= VOCAB_SIZE)
        self.decoder_embedding = nn.Embedding(VOCAB_SIZE + 1, n_embd , padding_idx= VOCAB_SIZE)
        self.positional_encoding = PositionalEncoding(n_embd, block_size)

        self.encoder_layers = nn.ModuleList([EncoderBlock(n_embd, n_head, dropout) for _ in range(n_layer)])
        self.decoder_layers = nn.ModuleList([DecoderBlock(n_embd, n_head, dropout) for _ in range(n_layer)])

        self.fc = nn.Linear(n_embd, VOCAB_SIZE + 1)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, src, tgt):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        seq_length = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length, device=device), diagonal=1)).bool()
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask

    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)

        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))

        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

        output = self.fc(dec_output)
        return output

