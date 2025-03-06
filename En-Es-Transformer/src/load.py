from bpemb import BPEmb
from transformer import Transformer
import torch
import config



def getTokenizer(lang : str = None , vs : int= None) -> BPEmb:
    '''Function to fetch the  BPemb tokenizer'''
    if (lang is None):
        raise ValueError("Language must be provided")
    if (vs is None or type(vs) != int):
        raise ValueError("Language must be provided")

    return BPEmb(lang = lang, vs = vs)


def loadTransformer() -> Transformer :
    model_path = "model/final/model_epoch_3.pth"
    state_dict = torch.load(model_path,weights_only=True)
    transformer_loaded = Transformer().to(config.device)
    transformer_loaded.load_state_dict(state_dict)
    transformer_loaded.eval()
    return transformer_loaded
