# import pandas as pd
# from simpletransformers.classification import ClassificationModel
# from sklearn.metrics import classification_report
# import yaml
# import os
# import sys
# import json

# sys.path.append("./scripts/")

# class Tester:
#     def __init__(self, config_path, test_data_path, use_cuda=True):
#         """
#         Initializes the Tester class.

#         :param config_path: Path to the YAML configuration file.
#         :param test_data_path: Path to the test data CSV file.
#         :param use_cuda: Boolean indicating whether to use GPU.
#         """
#         self.config_path = config_path
#         self.test_data_path = test_data_path
#         self.use_cuda = use_cuda
#         self.model = None
#         self.model_type = None
#         self.model_name = None
#         self.test_df = None
#         self.predictions = None
#         self.true_labels = None
#         self.label_mapping = None

#         self.load_config()
#         self.load_model()

#     def load_config(self):
#         """Loads the configuration file and sets model parameters."""
#         with open(self.config_path, 'r') as file:
#             config = yaml.safe_load(file)

#         # Extract model_type and model_name directly from config
#         models_config = config.get('models', {})
#         self.model_type = models_config.get('model_type')
#         self.model_name = models_config.get('model_name')

#         if not self.model_type or not self.model_name:
#             raise ValueError("Both 'model_type' and 'model_name' must be specified in the config file under 'models'.")

#         # Set other configurations if needed
#         trainer_config = config.get('trainer', {})
#         self.output_dir = trainer_config.get('output_dir', 'outputs/')
#         self.label_mapping_path = os.path.join(self.output_dir, 'label_mapping.json')

#     def load_model(self):
#         """Loads the trained model."""
#         # Determine the path to the best model
#         best_model_dir = os.path.join(self.output_dir, 'best_model')

#         if not os.path.exists(best_model_dir):
#             raise FileNotFoundError(f"Best model directory not found at '{best_model_dir}'.")

#         # Load the label mapping
#         if os.path.exists(self.label_mapping_path):
#             with open(self.label_mapping_path, 'r') as f:
#                 label_mapping_str_keys = json.load(f)
#                 # Convert string keys back to integers
#                 self.label_mapping = {int(k): v for k, v in label_mapping_str_keys.items()}
#         else:
#             raise FileNotFoundError(f"Label mapping file not found at '{self.label_mapping_path}'.")

#         # Initialize the model
#         self.model = ClassificationModel(
#             self.model_type,
#             best_model_dir,
#             use_cuda=self.use_cuda,
#             num_labels=len(self.label_mapping)
#         )

#     def load_test_data(self):
#         """Loads the test data from the specified path."""
#         self.test_df = pd.read_csv(self.test_data_path)

#         # Ensure that 'text' column exists
#         if 'text' not in self.test_df.columns:
#             raise ValueError("Test DataFrame must contain a 'text' column.")

#         # Handle labels
#         if 'labels' in self.test_df.columns:
#             # If labels are strings, map them using label_mapping
#             if self.test_df['labels'].dtype == object:
#                 inverse_label_mapping = {v: k for k, v in self.label_mapping.items()}
#                 self.test_df['labels'] = self.test_df['labels'].map(inverse_label_mapping)
#         elif 'label' in self.test_df.columns:
#             # Rename 'label' column to 'labels'
#             self.test_df = self.test_df.rename(columns={'label': 'labels'})
#             if self.test_df['labels'].dtype == object:
#                 inverse_label_mapping = {v: k for k, v in self.label_mapping.items()}
#                 self.test_df['labels'] = self.test_df['labels'].map(inverse_label_mapping)
#         else:
#             raise ValueError("Test DataFrame must contain a 'labels' or 'label' column.")

#     def make_predictions(self):
#         """Makes predictions on the test data."""
#         if self.test_df is not None:
#             self.predictions, _ = self.model.predict(self.test_df['text'].tolist())
#             self.true_labels = self.test_df['labels'].tolist()
#         else:
#             raise ValueError("Test data not loaded. Call load_test_data() first.")

#     def evaluate_model(self):
#         """Evaluates the model's predictions against the true labels."""
#         if self.predictions is not None and self.true_labels is not None:
#             report = classification_report(
#                 self.true_labels,
#                 self.predictions,
#                 target_names=[self.label_mapping[i] for i in sorted(self.label_mapping.keys())],
#                 zero_division=0
#             )
#             print(report)
#         else:
#             raise ValueError("Predictions not made. Call make_predictions() first.")

#     def run(self):
#         """Runs the testing process."""
#         self.load_test_data()
#         self.make_predictions()
#         self.evaluate_model()


# # Usage
# if __name__ == "__main__":
#     tester = Tester(
#         config_path='config.yml',                  # Path to your configuration file
#         test_data_path="./data/ground_truth_test.csv",  # Adjust the path to your test data
#         use_cuda=True                              # Set to False if not using GPU
#     )
#     tester.run()


import sys
import pandas as pd
from simpletransformers.classification import ClassificationModel
from sklearn.metrics import classification_report
import yaml
import os
import json
import warnings
import logging

# Suppress warnings and logging for cleaner output
warnings.simplefilter(action="ignore", category=FutureWarning)
logging.basicConfig(level=logging.ERROR)

sys.path.append("./scripts")
from utils import config, models_information  # Ensure this correctly imports your config function


class Tester:
    def __init__(self, config_path, test_data_path, use_cuda=True):
        """
        Initializes the Tester class.

        :param config_path: Path to the YAML configuration file.
        :param test_data_path: Path to the test data CSV file.
        :param use_cuda: Boolean indicating whether to use GPU.
        """
        self.config_path = config_path
        self.test_data_path = test_data_path
        self.use_cuda = use_cuda
        self.model = None
        self.model_type = None
        self.model_name = None
        self.test_df = None
        self.predictions = None
        self.true_labels = None
        self.label_mapping = None
        self.best_model_dir = None
        self.label_mapping_path = None

        self.load_config()
        self.load_model()

    def load_config(self):
        """Loads the configuration file and sets model parameters."""
        with open(self.config_path, 'r') as file:
            config_data = yaml.safe_load(file)

        # Extract model_type from models where the value is True
        models_config = config_data.get('models', {})
        self.model_type = None
        for key, value in models_config.items():
            if value is True:
                self.model_type = key
                break

        if not self.model_type:
            raise ValueError("No model set to True in the config file under 'models'.")

        # Set default model_name based on model_type
        try:
            model_name_defaults = models_information()
        except Exception as e:
            raise ValueError(f"Error loading model information: {e}".capitalize())

        self.model_name = model_name_defaults.get(self.model_type)

        if not self.model_name:
            raise ValueError(f"No default model_name found for model_type '{self.model_type}'. Please specify 'model_name' in the config file under 'models'.")

        # Set other configurations if needed
        trainer_config = config_data.get('trainer', {})
        self.output_dir = trainer_config.get('output_dir', 'outputs/')

        # Set the primary best model and label mapping paths
        self.best_model_dir = os.path.join(self.output_dir, 'best_model')
        self.label_mapping_path = os.path.join(self.output_dir, 'label_mapping.json')

        # Read the best fold number to potentially update paths
        best_fold_file = os.path.join(self.output_dir, 'best_fold.txt')
        if os.path.exists(best_fold_file):
            with open(best_fold_file, 'r') as f:
                best_fold = f.read().strip()
                # Check if the best fold directory exists
                best_fold_model_dir = os.path.join(self.output_dir, f'fold_{best_fold}', 'best_model')
                best_fold_label_mapping_path = os.path.join(self.output_dir, f'fold_{best_fold}', 'results', 'label_mapping.json')
                
                # Override primary paths if fold-specific best model exists
                if os.path.exists(best_fold_model_dir) and os.path.exists(best_fold_label_mapping_path):
                    self.best_model_dir = best_fold_model_dir
                    self.label_mapping_path = best_fold_label_mapping_path
        else:
            print(f"Warning: '{best_fold_file}' not found. Defaulting to 'outputs/best_model' directory.")

    def load_model(self):
        """Loads the trained model."""
        print(f"Attempting to load model from: {self.best_model_dir}")
        print(f"Looking for label mapping at: {self.label_mapping_path}")

        # Ensure the best model directory exists
        if not os.path.exists(self.best_model_dir):
            raise FileNotFoundError(f"Best model directory not found at '{self.best_model_dir}'.")

        # Load the label mapping
        if os.path.exists(self.label_mapping_path):
            with open(self.label_mapping_path, 'r') as f:
                label_mapping_str_keys = json.load(f)
                # Convert string keys back to integers
                self.label_mapping = {int(k): v for k, v in label_mapping_str_keys.items()}
        else:
            raise FileNotFoundError(f"Label mapping file not found at '{self.label_mapping_path}'.")

        # Initialize the model
        self.model = ClassificationModel(
            self.model_type,
            self.best_model_dir,
            use_cuda=self.use_cuda,
            num_labels=len(self.label_mapping)
        )

    def load_test_data(self):
        """Loads the test data from the specified path."""
        self.test_df = pd.read_csv(self.test_data_path)

        # Ensure that 'text' column exists
        if 'text' not in self.test_df.columns:
            raise ValueError("Test DataFrame must contain a 'text' column.")

        # Handle the label column
        if 'label' in self.test_df.columns:
            # If labels are strings, map them using label_mapping
            if self.test_df['label'].dtype == object:
                inverse_label_mapping = {v: k for k, v in self.label_mapping.items()}
                self.test_df['label'] = self.test_df['label'].map(inverse_label_mapping)
        else:
            raise ValueError("Test DataFrame must contain a 'label' column.")

    def make_predictions(self):
        """Makes predictions on the test data."""
        if self.test_df is not None:
            self.predictions, _ = self.model.predict(self.test_df['text'].tolist())
            self.true_labels = self.test_df['label'].tolist()
        else:
            raise ValueError("Test data not loaded. Call load_test_data() first.")

    def evaluate_model(self):
        """Evaluates the model's predictions against the true labels."""
        if self.predictions is not None and self.true_labels is not None:
            report = classification_report(
                self.true_labels,
                self.predictions,
                target_names=[self.label_mapping[i] for i in sorted(self.label_mapping.keys())],
                zero_division=0
            )
            print("Model evaluation completed.")
            print(report)
        else:
            raise ValueError("Predictions not made. Call make_predictions() first.")

    def run(self):
        """Runs the testing process."""
        self.load_test_data()
        self.make_predictions()
        self.evaluate_model()


# Usage
if __name__ == "__main__":
    tester = Tester(
        config_path='config.yml',                 # Path to your configuration file
        test_data_path="./data/processed/processed_test.csv",  # Adjust the path to your test data
        use_cuda=True                             # Set to False if not using GPU
    )
    tester.run()
