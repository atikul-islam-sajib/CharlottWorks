import sys
import yaml
import torch

sys.path.append("./scripts/")

def config():
    with open("./config.yml", "r") as file:
        return yaml.safe_load(file)
    
    
def device_init(self, device: str = "cuda"):
    if device == "cuda" and torch.cuda.is_available():
        self.device = torch.device("cuda")
    else:
        self.device = torch.device("cpu")