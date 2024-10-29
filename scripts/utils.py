import sys
import yaml
import torch

sys.path.append("./scripts/")

def config():
    with open("./config.yml", "r") as file:
        return yaml.safe_load(file)
    
    
def device_init(device: str = "cuda"):
    if device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")