import pandas as pd
import numpy as np
import os
import json
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from simpletransformers.classification import ClassificationModel, ClassificationArgs
import argparse
import sys

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

    def train(self, df_train, df_test=None):
        # Ensure that df_train contains 'text' and 'label' columns
        assert (
            "text" in df_train.columns and "label" in df_train.columns
        ), "df_train must contain 'text' and 'label' columns."

        # Encode labels if they are not integers
        if df_train["label"].dtype not in [np.int64, np.int32]:
            df_train["label"], uniques = pd.factorize(df_train["label"])
            label_mapping = dict(enumerate(uniques))
        else:
            labels = df_train["label"].unique()
            labels.sort()
            label_mapping = {label: str(label) for label in labels}

        self.label_mapping = label_mapping
        self.num_labels = len(label_mapping)

        # Create results directory
        os.makedirs("results", exist_ok=True)

        if self.use_kfold:
            self._kfold_training(df_train)
        else:
            self._normal_training(df_train, df_test)

        # Save label mapping
        with open("results/label_mapping.json", "w") as f:
            json.dump(self.label_mapping, f)

    def _kfold_training(self, df_train):
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        all_predictions = []
        all_true_labels = []
        fold = 1

        best_score = 0
        best_model = None

        for train_index, val_index in kf.split(df_train):
            print(f"Fold {fold}")
            train_data = df_train.iloc[train_index]
            val_data = df_train.iloc[val_index]

            # Update output directory for each fold
            self.model_args.output_dir = f"outputs/fold_{fold}/"

            # Create a ClassificationModel
            model = ClassificationModel(
                "roberta",
                "roberta-base",
                num_labels=self.num_labels,
                args=self.model_args,
                use_cuda=self.use_cuda,
            )

            # Train the model
            model.train_model(train_data[["text", "label"]])

            # Predict on validation set
            val_texts = val_data["text"].tolist()
            val_labels = val_data["label"].tolist()
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
                val_labels, predictions, target_names=self.label_mapping.values()
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
            all_true_labels, all_predictions, target_names=self.label_mapping.values()
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

    def _normal_training(self, df_train, df_test):
        # Update output directory
        self.model_args.output_dir = self.model_args.output_dir

        # Create a ClassificationModel
        model = ClassificationModel(
            "roberta",
            "roberta-base",
            num_labels=self.num_labels,
            args=self.model_args,
            use_cuda=self.use_cuda,
        )

        # Train the model
        model.train_model(df_train[["text", "label"]])

        # Save the model
        model.save_model("best_model/")

        # Predict on test set
        if df_test is not None:
            assert (
                "text" in df_test.columns and "label" in df_test.columns
            ), "df_test must contain 'text' and 'label' columns."
            test_texts = df_test["text"].tolist()
            test_labels = df_test["label"].tolist()
            predictions, raw_outputs = model.predict(test_texts)

            # Save classification report
            report = classification_report(
                test_labels, predictions, target_names=self.label_mapping.values()
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
            cm, index=self.label_mapping.values(), columns=self.label_mapping.values()
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
        action='store_true',
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
        action='store_true',
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

    trainer.train(df_train, df_test)
    trainer.save_model_args()
