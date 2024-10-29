import os
import pandas as pd
import re


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

        # Load data
        self.df_train = pd.read_csv(self.train_path)
        self.df_test = pd.read_csv(self.test_path)

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
        # Apply bracket extraction to relevant columns
        for col in ["event_trigger", "keyword"]:
            self.df_train[col] = self.df_train[col].apply(
                self.extract_content_in_brackets
            )
            self.df_test[col] = self.df_test[col].apply(
                self.extract_content_in_brackets
            )

        # Select relevant columns
        self.df_train = self.df_train[
            ["document", "text", "event_trigger", "keyword", "category"]
        ]
        self.df_test = self.df_test[
            ["document", "text", "event_trigger", "keyword", "category"]
        ]

    def format_text_columns(self):
        def format_text(row):
            sentence = row["text"].replace(
                row["keyword"], f"[KEYWORD] {row['keyword']} [/KEYWORD]"
            )
            sentence = sentence.replace(
                row["event_trigger"], f"[TRIGGER] {row['event_trigger']} [/TRIGGER]"
            )
            return sentence

        # Apply formatting
        self.df_train["text"] = self.df_train.apply(format_text, axis=1)
        self.df_test["text"] = self.df_test.apply(format_text, axis=1)

    def rename_and_map_labels(self):
        # Rename "category" to "label" and map labels to integers
        self.df_train = self.df_train.rename(columns={"category": "label"})
        self.df_test = self.df_test.rename(columns={"category": "label"})

        self.df_train["label"] = self.df_train["label"].map(self.label_mapping)
        self.df_test["label"] = self.df_test["label"].map(self.label_mapping)

    def save_processed_data(self):
        # Save processed datasets to the output directory
        train_output_path = os.path.join(self.output_dir, "processed_train.csv")
        test_output_path = os.path.join(self.output_dir, "processed_test.csv")

        self.df_train.to_csv(train_output_path, index=False)
        self.df_test.to_csv(test_output_path, index=False)
        print(f"Processed training data saved to {train_output_path}")
        print(f"Processed test data saved to {test_output_path}")

    def get_data_splits(self):
        # Returns data splits for further use
        X_train = self.df_train["text"]
        y_train = self.df_train["label"]
        X_test = self.df_test["text"]
        y_test = self.df_test["label"]
        return X_train, y_train, X_test, y_test

    def run_preprocessing(self):
        self.preprocess_columns()
        self.format_text_columns()
        self.rename_and_map_labels()
        self.save_processed_data()
        print("Data preprocessing complete.")


if __name__ == "__main__":
    pass
