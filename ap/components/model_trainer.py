import os.path
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import plotnine as pn
import torch
import torch.nn as nn
from torch import Tensor
from torchmetrics.classification import Accuracy, Precision, Recall, F1Score

from ap.ap_exception import APException
from ap.ap_logger import logging
from ap.components.data_tokenizer import DataTokenizer
from ap.constants import ACTION_NB_COLS, ModelName
from ap.entity.artifact_entity import (
    DataToSequenceArtifact, ModelTrainerArtifact, DataTokenizerArtifact,
    MetricArtifact, ModelMetrics, DataMergingArtifact
)
from ap.entity.config_entity import ModelTrainerConfig
from ap.utils.data_loader import DataLoader
from ap.utils.main_utils import read_data, pad_sequence


class ModelTrainer:
    scheduler = None

    def __init__(
            self,
            model,
            data_merging_artifact: DataMergingArtifact,
            model_trainer_config: ModelTrainerConfig,
            data_tokenizer_artifact: DataTokenizerArtifact,
            data_to_sequence_artifact: DataToSequenceArtifact,
    ):
        self.model_trainer_config = model_trainer_config
        self.data_to_sequence_artifact = data_to_sequence_artifact

        # Load the tokenizer
        self.tokenizer = DataTokenizer(data_merging_artifact=data_merging_artifact)
        self.tokenizer.load(data_tokenizer_artifact.tokenizer_file_path)

        self.model = model

        self.device = model_trainer_config.device
        self.name_vocab_size = model_trainer_config.name_vocab_size
        self.pad_token_idx = model_trainer_config.pad_token_idx
        self.action_start_idx = data_merging_artifact.household_columns_number + data_merging_artifact.person_columns_number
        self.max_sequence_length = self.action_start_idx + data_merging_artifact.trip_columns_number

        # Best model path. Create parent directory if it does not exist
        self.best_model_path = model_trainer_config.best_model_path
        os.makedirs(Path(self.best_model_path).parent, exist_ok=True)

        # Optimization
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=3e-4, betas=(0.9, 0.95))
        self.criterion = nn.CrossEntropyLoss()
        self.criterion.to(model_trainer_config.device)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda step: 0.85 ** (step // 10))

        self.losses = {'train': [], 'test': []}
        self.best_loss = float('inf')

    def create_data_loader(self, X: np.ndarray, y: np.ndarray, shuffle=True) -> DataLoader:
        """
        This method of ModelTrainer class is responsible for creating data loader
        """
        try:
            logging.info("Create data loader")
            return DataLoader(X, y, batch_size=self.model_trainer_config.batch_size, shuffle=shuffle)
        except Exception as e:
            raise APException(e, sys)

    def value_to_vector(self, value) -> Tuple[Tensor, Tensor, Tensor]:
        y1 = torch.eye(self.name_vocab_size['action']).to(self.device)[value[:, 0].long()]
        y2 = torch.eye(self.name_vocab_size['duration']).to(self.device)[value[:, 1].long()]
        y3 = torch.eye(self.name_vocab_size['distance']).to(self.device)[value[:, 2].long()]

        return y1, y2, y3

    def save_best_model(self) -> None:
        torch.save(self.model.state_dict(), self.best_model_path)

    def load_best_model(self) -> None:
        self.model.load_state_dict(torch.load(self.best_model_path, map_location=self.device))

    def save_model(self, path) -> None:
        torch.save(self.model.state_dict(), path)

    def load_model(self, path) -> None:
        self.model.load_state_dict(torch.load(path))

    def loss_fn(self, y_hat: Tuple[Tensor, Tensor, Tensor], y: Tuple[Tensor, Tensor, Tensor]):
        return (
                self.criterion(y_hat[0], y[0]) +
                self.criterion(y_hat[1], y[1]) +
                self.criterion(y_hat[2], y[2])
        )

    def forward(self,
                data_loader: DataLoader,
                is_training: bool = False,
                name: str = "Eval Loss",
                verbose: bool = False) -> Tensor:
        """
        Performs a forward pass through the model over the data provided by data_loader.

        Args:
            data_loader (DataLoader): The data loader providing input-output batches.
            is_training (bool): If True, runs training mode; otherwise, evaluation mode.
            name (str): Label used for logging the loss.
            verbose (bool): If True, prints the average loss.

        Returns:
            Tensor: The average loss over all batches.
        """

        self.model.train() if is_training else self.model.eval()
        total_loss = 0.0
        num_batches = 0

        context = torch.enable_grad if is_training else torch.no_grad
        with context():
            for X, y in data_loader:
                X, y = torch.tensor(X, device=self.device), torch.tensor(y, device=self.device)

                y = self.value_to_vector(y)

                y_hat = self.model(X)

                loss = self.loss_fn(y_hat, y)

                if is_training:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        if verbose:
            print(f"{name} : {avg_loss:.4f}")

        return torch.tensor(avg_loss, device=self.device)

    def train(self,
              train_loader: DataLoader,
              test_dataloader: DataLoader,
              epochs: int = 10,
              verbose: bool = True
              ) -> Dict[str, list]:
        """
        Trains the model for a given number of epochs and tracks training and validation losses.

        Args:
            train_loader (DataLoader): Dataloader for training data.
            test_dataloader (DataLoader): Dataloader for validation/testing data.
            epochs (int): Number of training epochs.
            verbose (bool): If True, prints progress every 10 epochs.

        Returns:
            Dict[str, list]: Dictionary containing lists of training and test losses.
        """

        for epoch in range(epochs):
            train_loss = self.forward(train_loader, is_training=True, name="Train Loss", verbose=False)
            test_loss = self.forward(test_dataloader, is_training=False, name="Test Loss", verbose=False)

            # Store float values instead of tensors for easier processing/logging later
            self.losses['train'].append(float(train_loss))
            self.losses['test'].append(float(test_loss))

            # Save best model
            if test_loss < self.best_loss:
                self.best_loss = test_loss
                self.save_best_model()

            # Logging
            if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
                print(
                    f"Epoch {epoch + 1}/{epochs} | "
                    f"LR: {self.optimizer.param_groups[0]['lr']:.6f} | "
                    f"Train Loss: {float(train_loss):.4f} | "
                    f"Test Loss: {float(test_loss):.4f} | "
                    f"Best Loss: {float(self.best_loss):.4f}"
                )

            # Learning rate scheduling
            if self.scheduler is not None:
                self.scheduler.step()

        return self.losses

    def compute_metrics(self, y_true: Tensor, y_pred: Tensor, num_classes: int = 1) -> ModelMetrics:
        """
        Computes classification metrics (accuracy, precision, recall, F1) for the given predictions.

        Args:
            y_true (Tensor): Ground truth labels.
            y_pred (Tensor): Predicted labels.
            num_classes (int): Number of classes for classification (default is 1).

        Returns:
            ModelMetrics: Object containing accuracy, precision, recall, and F1 score.
        """
        try:
            # Ensure tensors are on the correct device
            y_true = y_true.to(self.device)
            y_pred = y_pred.to(self.device)

            # Shape validation
            if y_true.shape != y_pred.shape:
                raise ValueError(f"Shape mismatch: y_true={y_true.shape}, y_pred={y_pred.shape}")

            # Initialize metrics
            metrics_kwargs = {
                "task": "multiclass",
                "num_classes": num_classes,
                "average": "macro"
            }

            accuracy = Accuracy(**metrics_kwargs).to(self.device)
            precision = Precision(**metrics_kwargs).to(self.device)
            recall = Recall(**metrics_kwargs).to(self.device)
            f1 = F1Score(**metrics_kwargs).to(self.device)

            # Compute metrics
            return ModelMetrics(
                accuracy=accuracy(y_true, y_pred).item(),
                precision=precision(y_true, y_pred).item(),
                recall=recall(y_true, y_pred).item(),
                f1_score=f1(y_true, y_pred).item()
            )

        except Exception as e:
            raise APException(f"Failed to compute metrics: {str(e)}", sys) from e

    def evaluate(self, test_loader: DataLoader) -> MetricArtifact:
        """
        This method of ModelTrainer class is responsible for evaluating the model
        """
        try:
            logging.info("Evaluating the model")

            # Load the best model
            self.load_best_model()

            # Set the model to evaluation mode
            self.model.eval()
            y_trues = {  # noqa
                "action": torch.empty((0, self.name_vocab_size['action']), device=self.device),
                "duration": torch.empty((0, self.name_vocab_size['duration']), device=self.device),
                "distance": torch.empty((0, self.name_vocab_size['distance']), device=self.device)
            }
            y_preds = {  # noqa
                "action": torch.empty((0, self.name_vocab_size['action']), device=self.device),
                "duration": torch.empty((0, self.name_vocab_size['duration']), device=self.device),
                "distance": torch.empty((0, self.name_vocab_size['distance']), device=self.device)
            }

            for X, y in test_loader:
                X = X.to(self.device) if isinstance(X, Tensor) else torch.tensor(X, device=self.device)
                y = y.to(self.device) if isinstance(y, Tensor) else torch.tensor(y, device=self.device)

                y1, y2, y3 = self.value_to_vector(y)
                y_pred = self.model(X)

                y1_pred, y2_pred, y3_pred = y_pred[0], y_pred[1], y_pred[2]

                # Concatenate the tensors
                y_trues["action"] = torch.cat((y_trues["action"], y1), dim=0)  # noqa
                y_trues["duration"] = torch.cat((y_trues["duration"], y2), dim=0)
                y_trues["distance"] = torch.cat((y_trues["distance"], y3), dim=0)

                y_preds["action"] = torch.cat((y_preds["action"], y1_pred), dim=0)  # noqa
                y_preds["duration"] = torch.cat((y_preds["duration"], y2_pred), dim=0)
                y_preds["distance"] = torch.cat((y_preds["distance"], y3_pred), dim=0)

            # Calculate accuracy for each output using torchmetrics
            action_metrics = self.compute_metrics(
                torch.argmax(y_trues["action"], dim=1),
                torch.argmax(y_preds["action"], dim=1),
                num_classes=self.name_vocab_size['action']
            )

            duration_metrics = self.compute_metrics(
                torch.argmax(y_trues["duration"], dim=1),
                torch.argmax(y_preds["duration"], dim=1),
                num_classes=self.name_vocab_size['duration']
            )

            distance_metrics = self.compute_metrics(
                torch.argmax(y_trues["distance"], dim=1),
                torch.argmax(y_preds["distance"], dim=1),
                num_classes=self.name_vocab_size['distance']
            )

            # Create a MetricArtifact object
            return MetricArtifact(
                action_metrics=action_metrics,
                duration_metrics=duration_metrics,
                distance_metrics=distance_metrics,
                best_model_validation_loss=self.best_loss,
            )

        except Exception as e:
            raise APException(e, sys)

    def plot_losses(self) -> pn.ggplot:
        df_plot = pd.concat([
            pd.DataFrame(self.losses['train'], columns=['loss']).assign(type='train', epoch=np.arange(1, len(
                self.losses['train']) + 1)),
            pd.DataFrame(self.losses['test'], columns=['loss']).assign(type='test',
                                                                       epoch=np.arange(1, len(self.losses['test']) + 1))
        ])

        return (
                pn.ggplot(df_plot)
                + pn.aes(x='epoch', y='loss', color='type')  # noqa
                + pn.geom_line(size=1)
                + pn.scale_color_brewer(type='qualitative', palette=6,
                                        labels=lambda labs: [f"{x.title()}" for x in labs])
                + pn.labs(x="Epochs", y="Loss", color="")
                + pn.theme_538()
                + pn.theme(
            legend_position="top",
            figure_size=(12, 6)
        )
        )

    def generate_one_row(self, X,
                         temperature: int = 1.0,
                         do_sample: bool = False,
                         top_k: int = None
                         ) -> Tuple[np.ndarray, np.ndarray]:

        # Assure that X's shape is (1, N)
        activity_list = np.array([]).reshape(0, ACTION_NB_COLS)
        X = X[:1]

        test = self.pad_token_idx

        if self.model_trainer_config.model_name == ModelName.LSTM.value:
            X = pad_sequence(X, self.max_sequence_length, self.pad_token_idx, padding="pre")
        else:
            X = pad_sequence(X, self.max_sequence_length, self.pad_token_idx, padding="post")

        with torch.no_grad():
            for idx in range(self.action_start_idx, self.max_sequence_length, ACTION_NB_COLS):
                y1, y2, y3 = self.model(torch.tensor(X).to(self.device))

                y1_probs = y1 / temperature
                y2_probs = y2 / temperature
                y3_probs = y3 / temperature

                if top_k is not None:
                    top_y1 = torch.topk(y1_probs, top_k, dim=1)
                    top_y2 = torch.topk(y2_probs, top_k, dim=1)
                    top_y3 = torch.topk(y3_probs, top_k, dim=1)

                    # Replace the probabilities with the top k probabilities and set the rest to 0
                    y1_probs = torch.full_like(y1_probs, 0)
                    y2_probs = torch.full_like(y2_probs, 0)
                    y3_probs = torch.full_like(y3_probs, 0)

                    y1_probs.scatter_(1, top_y1.indices, top_y1.values)
                    y2_probs.scatter_(1, top_y2.indices, top_y2.values)
                    y3_probs.scatter_(1, top_y3.indices, top_y3.values)

                if do_sample:
                    y1 = torch.multinomial(y1_probs, num_samples=1).squeeze()
                    y2 = torch.multinomial(y2_probs, num_samples=1).squeeze()
                    y3 = torch.multinomial(y3_probs, num_samples=1).squeeze()
                else:
                    y1 = torch.argmax(y1_probs, dim=1)
                    y2 = torch.argmax(y2_probs, dim=1)
                    y3 = torch.argmax(y3_probs, dim=1)

                y1 = self.tokenizer.convert_index_from_name("action", y1.cpu().numpy())
                y2 = self.tokenizer.convert_index_from_name("duration", y2.cpu().numpy())
                y3 = self.tokenizer.convert_index_from_name("distance", y3.cpu().numpy())

                y_pred = np.array([y1, y2, y3]).reshape(1, -1)

                if self.model_trainer_config.model_name == ModelName.LSTM.value:
                    X = np.concatenate((X[:, ACTION_NB_COLS:], y_pred), axis=1)
                else:
                    X[:, idx:(idx + ACTION_NB_COLS)] = y_pred

                activity_list = np.concatenate((activity_list, y_pred), axis=0)

                # Check if y_pred contains pad_token break
                if np.isin(y_pred, self.pad_token_idx).any():
                    break


        # Check if the x ndim == 2 and the activity_list ndim == 2
        assert X.ndim == 2, "X should be 2 dimensional"
        assert activity_list.ndim == 2, "Y should be 2 dimensional"

        return X, activity_list

    def generate(self, X,
                 temperature: int = 1.0,
                 do_sample: bool = False,
                 top_k: int = None,
                 ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Function to generate the output sequence given the input sequence.
        It returns a tuple containing individual attributes and their corresponding sequence.

        Args :
            X : np.ndarray. Input sequence.
            output_path : str. Path to save the output sequence.
            temperature : int. Temperature for sampling. Default is 1.0.
            do_sample : bool. Whether to sample from the distribution. Default is False.
            top_k : int. Top k for sampling. Default is None.

        Returns :
            tuple : (X, Sequence)
        """
        try:
            logging.info("Generate output sequence")

            individual_attributes = list()
            activities_list = list()

            for idx in range(X.shape[0]):
                _, activities = self.generate_one_row(X[idx:idx + 1], temperature, do_sample, top_k)
                attributes = [idx, *X[idx].tolist()]
                activities = np.concatenate((
                    np.repeat(idx, activities.shape[0]).reshape(-1, 1),
                    np.arange(activities.shape[0]).reshape(-1, 1),
                    activities
                ), axis=1)
                individual_attributes.append(attributes)
                activities_list.append(activities)

            return np.array(individual_attributes), np.concatenate(activities_list, axis=0)
        except Exception as e:
            raise APException(e, sys)

    def initiate_training(self) -> ModelTrainerArtifact:
        """
        This method of ModelTrainer class is responsible for initiating the training
        """
        try:
            logging.info("Load datasets")
            x_train = read_data(self.data_to_sequence_artifact.train_x_data_as_sequence_file_path)
            y_train = read_data(self.data_to_sequence_artifact.train_y_data_as_sequence_file_path)
            x_validation = read_data(self.data_to_sequence_artifact.validation_x_data_as_sequence_file_path)
            y_validation = read_data(self.data_to_sequence_artifact.validation_y_data_as_sequence_file_path)

            logging.info("Create data loaders")
            train_loader = self.create_data_loader(x_train, y_train, shuffle=True)
            test_loader = self.create_data_loader(x_validation, y_validation, shuffle=False)

            logging.info("Start training")
            self.train(
                train_loader=train_loader,
                test_dataloader=test_loader,
                epochs=self.model_trainer_config.epochs,
                verbose=self.model_trainer_config.verbose
            )

            logging.info("Evaluate the model")
            metric_artifact = self.evaluate(test_loader)

            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.best_model_path,
                metric_artifact=metric_artifact,
                model_trainer_config=self.model_trainer_config.to_json(),
                model_name=self.model_trainer_config.model_name,
            )

            return model_trainer_artifact
        except Exception as e:
            raise APException(e, sys)
