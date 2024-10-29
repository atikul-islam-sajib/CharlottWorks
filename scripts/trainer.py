from simpletransformers.classification import ClassificationModel, ClassificationArgs
import pandas as pd
import re
from sklearn.model_selection import GroupShuffleSplit
import torch
from sklearn.metrics import (
    mean_squared_error,
    f1_score,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
import optuna
import seaborn as sns
import matplotlib.pyplot as plt
import os
import json


class Trainer:
    def __init__(
        self,
        num_train_epochs: int = 5,
        learning_rate: float = 8.521786512851659e-05,
        train_batch_size: int = 8,
        eval_batch_size: int = 24,
        max_seq_length: int = 512,
        output_dir: str = "outputs/",
        overwrite_output_dir: bool = True,
    ):
        self.num_train_epochs = num_train_epochs
        self.learning_rate = learning_rate
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.max_seq_length = max_seq_length
        self.output_dir = output_dir
        self.overwrite_output_dir = overwrite_output_dir
