# src/eval/utils_load_orm.py
import torch
from transformers import AutoTokenizer

from src.models.orm_scorer import ORMScorer

def load_orm(checkpoint, base_model, device):
    model = ORMScorer(base_model).to(device)
    ckpt = torch.load(checkpoint, map_location=device)

    state = ckpt["model_state"] if "model_state" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    return model

def load_tokenizer(base_model):
    tok = AutoTokenizer.from_pretrained(base_model)
    tok.padding_side = "left"
    tok.truncation_side = "left"
    return tok