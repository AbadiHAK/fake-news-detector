import json , random 
from pathlib import Path
import numpy as np
import torch


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w' , encoding = 'utf-8') as f:
        json.dump(obj , f,ensure_ascii=False , indent=2)


def load_json(path: Path):
    with open(path , 'r' , encoding="utf-8") as f:
        return json.load(f)
    
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"