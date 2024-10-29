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


def models_information():
    return {
        "albert": "albert-base-v2",
        "bert": "bert-base-uncased",
        "bertweet": "vinai/bertweet-base",
        "bigbird": "google/bigbird-roberta-base",
        "camembert": "camembert-base",
        "deberta": "microsoft/deberta-base",
        "distilbert": "distilbert-base-uncased",
        "electra": "google/electra-base-discriminator",
        "flaubert": "flaubert/flaubert_base_cased",
        "herbert": "allegro/herbert-base-cased",
        "layoutlm": "microsoft/layoutlm-base-uncased",
        "layoutlmv2": "microsoft/layoutlmv2-base-uncased",
        "longformer": "allenai/longformer-base-4096",
        "mpnet": "microsoft/mpnet-base",
        "rembert": "google/rembert",
        "roberta": "roberta-base",
        "squeezebert": "squeezebert/squeezebert-uncased",
        "xlm": "xlm-mlm-17-1280",
        "xlmroberta": "xlm-roberta-base",
        "xlnet": "xlnet-base-cased",
    }
