import os
import re
import sys
import argparse
import pandas as pd
from tqdm import tqdm

sys.path.append("./scripts")

from utils import config


class DataProcessor:
    def __init__(self, train_path, test_path, output_dir, label_mapping=None):
        self.train_path = train_path
        self.test_path = test_path
        self.output_dir = output_dir
        self.label_mapping = (
            label_mapping
            if label_mapping
            else {"action": 0, "intention": 1, "belief": 2, "situation": 3}
        )

        # Create the output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

        # Load data with error handling
        try:
            self.df_train = pd.read_csv(self.train_path)
            self.df_test = pd.read_csv(self.test_path)
        except FileNotFoundError as e:
            print(f"Error: {e}. Please check the file paths.")
            sys.exit(1)

    @staticmethod
    def extract_content_in_brackets(text):
        """
        Extracts and returns the content within the first pair of square brackets found in the given text.
        If no square brackets are found, returns an empty string.
        """
        text = str(text)
        match = re.search(r"\[(.*?)\]", text)
        return match.group(1) if match else ""

    def preprocess_columns(self):
        try:
            # Apply bracket extraction to relevant columns with progress visualization
            for col in ["event_trigger", "keyword"]:
                tqdm.pandas(desc=f"Extracting content in brackets for {col}")
                self.df_train[col] = self.df_train[col].progress_apply(
                    self.extract_content_in_brackets
                )
                self.df_test[col] = self.df_test[col].progress_apply(
                    self.extract_content_in_brackets
                )

            # Select relevant columns
            self.df_train = self.df_train[
                ["document", "text", "event_trigger", "keyword", "category"]
            ]
            self.df_test = self.df_test[
                ["document", "text", "event_trigger", "keyword", "category"]
            ]
        except KeyError as e:
            print(f"Error in preprocessing columns: {e}")
            sys.exit(1)

    def format_text_columns(self):
        try:
            def format_text(row):
                sentence = row["text"].replace(
                    row["keyword"], f"[KEYWORD] {row['keyword']} [/KEYWORD]"
                )
                sentence = sentence.replace(
                    row["event_trigger"], f"[TRIGGER] {row['event_trigger']} [/TRIGGER]"
                )
                return sentence

            # Apply formatting with progress bar
            tqdm.pandas(desc="Formatting text columns")
            self.df_train["text"] = self.df_train.progress_apply(format_text, axis=1)
            self.df_test["text"] = self.df_test.progress_apply(format_text, axis=1)
        except Exception as e:
            print(f"Error in formatting text columns: {e}")
            sys.exit(1)

    def rename_and_map_labels(self):
        try:
            # Rename "category" to "label" and map labels to integers
            self.df_train = self.df_train.rename(columns={"category": "label"})
            self.df_test = self.df_test.rename(columns={"category": "label"})

            self.df_train["label"] = self.df_train["label"].map(self.label_mapping)
            self.df_test["label"] = self.df_test["label"].map(self.label_mapping)
        except KeyError as e:
            print(f"Error in renaming or mapping labels: {e}")
            sys.exit(1)

    def save_processed_data(self):
        try:
            # Save processed datasets to the output directory
            train_output_path = os.path.join(self.output_dir, "processed_train.csv")
            test_output_path = os.path.join(self.output_dir, "processed_test.csv")

            self.df_train.to_csv(train_output_path, index=False)
            self.df_test.to_csv(test_output_path, index=False)
            print(f"Processed training data saved to {train_output_path}")
            print(f"Processed test data saved to {test_output_path}")
        except IOError as e:
            print(f"Error saving processed data: {e}")
            sys.exit(1)

    def get_data_splits(self):
        # Returns data splits for further use
        X_train = self.df_train["text"]
        y_train = self.df_train["label"]
        X_test = self.df_test["text"]
        y_test = self.df_test["label"]
        return X_train, y_train, X_test, y_test

    def run_preprocessing(self):
        try:
            self.preprocess_columns()
            self.format_text_columns()
            self.rename_and_map_labels()
            self.save_processed_data()
            print("Data preprocessing complete.")
        except Exception as e:
            print(f"An error occurred during preprocessing: {e}")
            sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Data Preprocessing for the SDG-Event-Classification Task".title()
    )
    parser.add_argument(
        "--train_path",
        type=str,
        default=config()["path"]["train_path"],
        help="Path to the training data".capitalize(),
    )
    parser.add_argument(
        "--test_path",
        type=str,
        default=config()["path"]["test_path"],
        help="Path to the test data".capitalize(),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=config()["path"]["output_dir"],
        help="Path to save the processed data".capitalize(),
    )
    
    args = parser.parse_args()
    
    try:
        processor = DataProcessor(
            train_path=args.train_path,
            test_path=args.test_path,
            output_dir=args.output_dir,
        )
        processor.run_preprocessing()
        
        X_train, y_train, X_test, y_test = processor.get_data_splits()
        print("Data splits created successfully.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)
