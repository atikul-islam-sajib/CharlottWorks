from simpletransformers.classification import ClassificationModel, ClassificationArgs
import pandas as pd
import re
from sklearn.model_selection import GroupShuffleSplit
import torch
from sklearn.metrics import mean_squared_error, f1_score, accuracy_score, classification_report, confusion_matrix
import optuna
import seaborn as sns
import matplotlib.pyplot as plt
import os
import json

class Loader():
    def __init__(self):
        pass