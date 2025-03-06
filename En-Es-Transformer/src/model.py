import config
from load import getTokenizer, loadTransformer
import torch
import torch.nn as  nn


class Translator:
    def __init__(self):
        self.en_tokenizer = getTokenizer(config.tokenizer_config['src_lang'] , config.tokenizer_config['src_vocab_size'])
        self.es_tokenizer = getTokenizer(config.tokenizer_config['tgt_lang'] , config.tokenizer_config['tgt_vocab_size'])
        self.transformer  = loadTransformer()
        self.BOS_IDX = self.en_tokenizer.BOS
        self.EOS_IDX = self.en_tokenizer.EOS
        self.PAD_IDX = self.en_tokenizer.vs
        self.UNK_IDX = 0
        self.VOCAB_SIZE = self.en_tokenizer.vs

    def restore_word_boundaries(self,tokens):
        """Restores word boundaries from tokenized IDs and replaces `_` with spaces."""
        decoded_tokens = [self.es_tokenizer.emb.index_to_key[i] for i in tokens if i < self.es_tokenizer.vs]
        return "".join(decoded_tokens).replace("▁"," ").strip()


    def translate(self,src):
        src_tokens = self.en_tokenizer.encode_ids_with_bos_eos(src)
        tgt_tokens = [self.BOS_IDX]

        src_vectors = torch.tensor((src_tokens + [self.PAD_IDX] * (config.block_size - len(src_tokens)))[:config.block_size], dtype=torch.long, device=config.device).unsqueeze(0)

        for i in range(config.block_size):
            tgt_vectors = torch.tensor((tgt_tokens + [self.PAD_IDX] * (config.block_size - len(tgt_tokens)))[:config.block_size], dtype=torch.long, device=config.device).unsqueeze(0)
            output = self.transformer(src_vectors, tgt_vectors)
            idx = torch.argmax(nn.functional.softmax(output, dim=2)[0][i]).item()
            tgt_tokens.append(idx)

            if idx == self.EOS_IDX:
                break

        return self.restore_word_boundaries(tgt_tokens[1 : -1])

