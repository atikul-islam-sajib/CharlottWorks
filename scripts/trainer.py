# Import necessary libraries
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from simpletransformers.classification import ClassificationModel, ClassificationArgs


import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


sys.path.append("./scripts")
from utils import config, device_init


class Trainer:
    def __init__(
        self,
        num_train_epochs=5,
        learning_rate=8.521786512851659e-05,
        train_batch_size=8,
        eval_batch_size=24,
        max_seq_length=512,
        output_dir="outputs/",
        overwrite_output_dir=True,
        device="cuda",
        use_kfold=False,
        n_splits=5,
        random_state=42,
        model_type="roberta"
    ):
        self.model_args = ClassificationArgs()
        self.model_args.num_train_epochs = num_train_epochs
        self.model_args.learning_rate = learning_rate
        self.model_args.train_batch_size = train_batch_size
        self.model_args.eval_batch_size = eval_batch_size
        self.model_args.max_seq_length = max_seq_length
        self.model_args.output_dir = output_dir
        self.model_args.overwrite_output_dir = overwrite_output_dir

        self.device = device
        self.use_cuda = device_init(device=device)
        self.use_kfold = use_kfold
        self.n_splits = n_splits
        self.random_state = random_state
        self.model_type = model_type
        
        
    def select_the_model(self):
        if self.model_type == config()["models"]["albert"]:
            return {"type": "albert", "base": "albert-base-v2"}  # Confirmed: "albert-base-v2" is typical
        elif self.model_type == config()["models"]["bert"]:
            return {"type": "bert", "base": "bert-base-uncased"}  # Confirmed: "bert-base-uncased" is typical
        elif self.model_type == config()["models"]["bertweet"]:
            return {"type": "bertweet", "base": "vinai/bertweet-base"}  # Confirmed
        elif self.model_type == config()["models"]["bigbird"]:
            return {"type": "bigbird", "base": "google/bigbird-roberta-base"}  # Confirmed
        elif self.model_type == config()["models"]["camembert"]:
            return {"type": "camembert", "base": "camembert-base"}  # Confirmed
        elif self.model_type == config()["models"]["deberta"]:
            return {"type": "deberta", "base": "microsoft/deberta-base"}  # Confirmed
        elif self.model_type == config()["models"]["distilbert"]:
            return {"type": "distilbert", "base": "distilbert-base-uncased"}  # Confirmed
        elif self.model_type == config()["models"]["electra"]:
            return {"type": "electra", "base": "google/electra-base-discriminator"}  # Confirmed
        elif self.model_type == config()["models"]["flaubert"]:
            return {"type": "flaubert", "base": "flaubert/flaubert-base-cased"}  # Confirmed
        elif self.model_type == config()["models"]["herbert"]:
            return {"type": "herbert", "base": "allegro/herbert-klej-cased-tokenizer-v1"}  # Verified name is "allegro/herbert-klej-cased-tokenizer-v1"
        elif self.model_type == config()["models"]["layoutlm"]:
            return {"type": "layoutlm", "base": "microsoft/layoutlm-base-uncased"}  # Confirmed
        elif self.model_type == config()["models"]["layoutlmv2"]:
            return {"type": "layoutlmv2", "base": "microsoft/layoutlmv2-base-uncased"}  # Confirmed
        elif self.model_type == config()["models"]["longformer"]:
            return {"type": "longformer", "base": "allenai/longformer-base-4096"}  # Confirmed
        elif self.model_type == config()["models"]["mpnet"]:
            return {"type": "mpnet", "base": "microsoft/mpnet-base"}  # Confirmed
        elif self.model_type == config()["models"]["mobilebert"]:
            return {"type": "mobilebert", "base": "google/mobilebert-uncased"}  # Confirmed
        elif self.model_type == config()["models"]["rembert"]:
            return {"type": "rembert", "base": "google/rembert"}  # Confirmed
        elif self.model_type == config()["models"]["roberta"]:
            return {"type": "roberta", "base": "roberta-base"}  # Confirmed
        elif self.model_type == config()["models"]["squeezebert"]:
            return {"type": "squeezebert", "base": "squeezebert/squeezebert-uncased"}  # Confirmed: "squeezebert/squeezebert-uncased"
        elif self.model_type == config()["models"]["xlm"]:
            return {"type": "xlm", "base": "xlm-roberta-base"}  # Confirmed
        elif self.model_type == config()["models"]["xlmroberta"]:
            return {"type": "xlmroberta", "base": "xlm-roberta-base"}  # Confirmed
        elif self.model_type == config()["models"]["xlnet"]:
            return {"type": "xlnet", "base": "xlnet-base-cased"}  # Confirmed
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        

    def train(self, df_train, df_test=None):
        # Ensure that df_train contains 'text' and 'labels' columns
        assert (
            "text" in df_train.columns and "labels" in df_train.columns
        ), "df_train must contain 'text' and 'labels' columns."

        # Encode labels if they are not integers
        if df_train["labels"].dtype not in [np.int64, np.int32, int]:
            df_train["labels"], uniques = pd.factorize(df_train["labels"])
            label_mapping = dict(enumerate(uniques))
        else:
            labels = df_train["labels"].unique()
            labels.sort()
            label_mapping = {label: str(label) for label in labels}

        self.label_mapping = label_mapping
        self.num_labels = len(label_mapping)

        # Create results directory
        os.makedirs("results", exist_ok=True)

        if self.use_kfold:
            self._kfold_training(df_train, df_test)
        else:
            self._normal_training(df_train, df_test)

        # Save label mapping with string keys
        label_mapping_str_keys = {str(k): v for k, v in self.label_mapping.items()}
        with open("results/label_mapping.json", "w") as f:
            json.dump(label_mapping_str_keys, f)

    def _kfold_training(self, df_train, df_test):
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        all_predictions = []
        all_true_labels = []
        fold = 1

        best_score = 0
        best_model = None

        for train_index, val_index in kf.split(df_train):
            print(f"\n\n############################ Fold - {fold} ############################ \n\n")
            train_data = df_train.iloc[train_index].reset_index(drop=True)
            val_data = df_train.iloc[val_index].reset_index(drop=True)

            # Update output directory for each fold
            self.model_args.output_dir = f"outputs/fold_{fold}/"

            # Create a ClassificationModel
            model = ClassificationModel(
                self.select_the_model()["type"],
                self.select_the_model()["base"],
                num_labels=self.num_labels,
                args=self.model_args,
                use_cuda=self.use_cuda,
            )

            # Train the model
            model.train_model(train_data)

            # Predict on validation set
            val_texts = val_data["text"].tolist()
            val_labels = val_data["labels"].tolist()
            predictions, raw_outputs = model.predict(val_texts)

            # Append results
            all_predictions.extend(predictions)
            all_true_labels.extend(val_labels)

            # Calculate validation score
            fold_score = accuracy_score(val_labels, predictions)

            # Save the model if it's the best so far
            if fold_score > best_score:
                best_score = fold_score
                best_model = model
                best_fold = fold

            # Save classification report
            report = classification_report(
                val_labels,
                predictions,
                target_names=list(self.label_mapping.values()),
                zero_division=0,
            )
            with open(f"results/classification_report_fold_{fold}.txt", "w") as f:
                f.write(report)

            # Plot confusion matrix
            self._plot_confusion_matrix(
                val_labels,
                predictions,
                f"Confusion Matrix - Fold {fold}",
                f"results/confusion_matrix_fold_{fold}",
            )

            fold += 1

        # Save overall classification report
        overall_report = classification_report(
            all_true_labels,
            all_predictions,
            target_names=list(self.label_mapping.values()),
            zero_division=0,
        )
        with open("results/classification_report_overall.txt", "w") as f:
            f.write(overall_report)

        # Plot overall confusion matrix
        self._plot_confusion_matrix(
            all_true_labels,
            all_predictions,
            "Confusion Matrix - Cross-Validation",
            "results/confusion_matrix_overall",
        )

        # Save the best model
        best_model.save_model("best_model/")
        print(f"The best model was from fold {best_fold} with a score of {best_score:.4f}")

        # Evaluate the best model on the test set
        if df_test is not None:
            self._evaluate_on_test_set(best_model, df_test)

    def _normal_training(self, df_train, df_test):
        # Update output directory
        self.model_args.output_dir = self.model_args.output_dir

        # Create a ClassificationModel
        model = ClassificationModel(
            self.select_the_model()["type"],
            self.select_the_model()["base"],
            num_labels=self.num_labels,
            args=self.model_args,
            use_cuda=self.use_cuda,
        )

        # Train the model
        model.train_model(df_train)

        # Save the model
        model.save_model("best_model/")

        # Predict on test set
        if df_test is not None:
            self._evaluate_on_test_set(model, df_test)

    def _evaluate_on_test_set(self, model, df_test):
        assert (
            "text" in df_test.columns and "labels" in df_test.columns
        ), "df_test must contain 'text' and 'labels' columns."

        # If labels are not numeric, map using the label_mapping
        if df_test["labels"].dtype not in [np.int64, np.int32, int]:
            df_test["labels"] = df_test["labels"].map(
                {v: int(k) for k, v in self.label_mapping.items()}
            )
            df_test["labels"] = df_test["labels"].fillna(-1).astype(int)
            if -1 in df_test["labels"].values:
                raise ValueError("Some labels in df_test are not in the training label mapping.")
        else:
            # Ensure labels are integers
            df_test["labels"] = df_test["labels"].astype(int)

        test_texts = df_test["text"].tolist()
        test_labels = df_test["labels"].tolist()
        predictions, raw_outputs = model.predict(test_texts)

        # Save classification report
        report = classification_report(
            test_labels,
            predictions,
            target_names=list(self.label_mapping.values()),
            zero_division=0,
        )
        with open("results/classification_report_test_set.txt", "w") as f:
            f.write(report)

        # Plot confusion matrix
        self._plot_confusion_matrix(
            test_labels,
            predictions,
            "Confusion Matrix - Test Set",
            "results/confusion_matrix_test_set",
        )

    def _plot_confusion_matrix(self, true_labels, predictions, title, filename):
        cm = confusion_matrix(true_labels, predictions)
        cm_df = pd.DataFrame(
            cm, index=list(self.label_mapping.values()), columns=list(self.label_mapping.values())
        )

        sns.set(context="paper", font_scale=1.7)
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm_df, annot=True, fmt="g", cmap="Blues")
        plt.title(title)
        plt.ylabel("Actual Labels")
        plt.xlabel("Predicted Labels")

        plt.savefig(f"{filename}.png", format="png", bbox_inches="tight")
        plt.savefig(f"{filename}.pdf", format="pdf", bbox_inches="tight")
        plt.close()

    def save_model_args(self):
        # Save model arguments to a JSON file
        model_args_dict = {
            "learning_rate": self.model_args.learning_rate,
            "num_train_epochs": self.model_args.num_train_epochs,
            "train_batch_size": self.model_args.train_batch_size,
            "eval_batch_size": self.model_args.eval_batch_size,
            "max_seq_length": self.model_args.max_seq_length,
        }

        with open("results/model_args.json", "w") as f:
            json.dump(model_args_dict, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Training Script")
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=config()["trainer"]["num_train_epochs"],
        help="Define the number of training epochs",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=config()["trainer"]["learning_rate"],
        help="Define the learning rate",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=config()["trainer"]["train_batch_size"],
        help="Define the training batch size",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=config()["trainer"]["eval_batch_size"],
        help="Define the evaluation batch size",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=config()["trainer"]["max_seq_length"],
        help="Define the maximum sequence length",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=config()["trainer"]["output_dir"],
        help="Define the output directory",
    )
    parser.add_argument(
        "--overwrite_output_dir",
        default=config()["trainer"]["overwrite_output_dir"],
        help="Overwrite the output directory",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=config()["trainer"]["device"],
        help="Define the device to use",
    )
    parser.add_argument(
        "--use_kfold",
        default=config()["trainer"]["use_kfold"],
        help="Use k-fold cross-validation",
    )
    parser.add_argument(
        "--n_splits",
        type=int,
        default=config()["trainer"]["n_splits"],
        help="Define the number of splits for k-fold cross-validation",
    )

    args = parser.parse_args()

    trainer = Trainer(
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        max_seq_length=args.max_seq_length,
        output_dir=args.output_dir,
        overwrite_output_dir=args.overwrite_output_dir,
        device=args.device,
        use_kfold=args.use_kfold,
        n_splits=args.n_splits,
    )

    # Load data
    df_train = pd.read_csv("./data/processed/processed_train.csv")
    df_test = pd.read_csv("./data/processed/processed_test.csv")

    # Rename 'label' column to 'labels'
    df_train = df_train.rename(columns={"label": "labels"})
    df_test = df_test.rename(columns={"label": "labels"})

    # Ensure that the DataFrames have the necessary columns
    assert "text" in df_train.columns and "labels" in df_train.columns, \
        "df_train must contain 'text' and 'labels' columns."
    assert "text" in df_test.columns and "labels" in df_test.columns, \
        "df_test must contain 'text' and 'labels' columns."

    trainer.train(df_train, df_test)
    trainer.save_model_args()
