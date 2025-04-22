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

from ap_gpt.ap_exception import APException
from ap_gpt.ap_logger import logging
from ap_gpt.components.data_tokenizer import DataTokenizer
from ap_gpt.entity.artifact_entity import DataToSequenceArtifact, ModelTrainerArtifact, DataTokenizerArtifact, \
    MetricArtifact, ModelMetrics, DataMergingArtifact
from ap_gpt.entity.config_entity import ModelTrainerConfig
from ap_gpt.utils.data_loader import DataLoader
from ap_gpt.utils.main_utils import read_data


class ModelTrainer:
    def __init__(
            self,
            model,
            data_merging_artifact: DataMergingArtifact,
            model_trainer_config : ModelTrainerConfig,
            data_tokenizer_artifact : DataTokenizerArtifact,
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
        self.max_sequence_length = model_trainer_config.max_sequence_length
        self.action_start_idx = model_trainer_config.action_start_idx

        # Best model path. Create parent directory if it does not exist
        self.best_model_path = model_trainer_config.best_model_path
        os.makedirs(Path(self.best_model_path).parent, exist_ok=True)


        # Optimization
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        self.criterion = nn.CrossEntropyLoss()
        self.criterion.to(model_trainer_config.device)
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=40, gamma=0.1)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda step: 0.85 ** (step // 10))

        self.losses = {'train': [], 'test': []}
        self.best_loss = float('inf')

    def create_data_loader(self, X : np.ndarray, y:np.ndarray, shuffle=True) -> DataLoader:
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
        self.model.load_state_dict(torch.load(self.best_model_path))

    def save_model(self, path) -> None:
        torch.save(self.model.state_dict(), path)

    def load_model(self, path) -> None:
        self.model.load_state_dict(torch.load(path))

    def forward(self,
                data_loader : DataLoader,
                is_training: bool = False,
                name: str = "Eval Loss",
                verbose: bool = False) -> Tensor:

        """
        This method of ModelTrainer class is responsible for performing forward pass

        Args:
            data_loader : DataLoader
            is_training : bool. If True, perform training. If False, perform evaluation.
            name : str. Name of the loss.
            verbose : bool. If True, print the loss.
        """
        if is_training:
            self.model.train()
        else:
            self.model.eval()

        losses = []
        for X, y in data_loader:

            X = torch.tensor(X).to(self.device)
            y = torch.tensor(y).to(self.device)


            y1, y2, y3 = self.value_to_vector(y)
            y_hat = self.model(X)
            loss = self.criterion(y_hat[0], y1) + self.criterion(y_hat[1], y2) + self.criterion(y_hat[2], y3)

            if is_training:
                self.model.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()

            losses.append(loss.item())

        loss = torch.mean(torch.tensor(losses))
        if verbose:
            print(f"{name} : {torch.mean(loss)}")
        return torch.mean(loss)

    def train(self,
              train_loader: DataLoader,
              test_dataloader : DataLoader,
              epochs=10,
              verbose: bool = True
        ) -> Dict[str, list]:
        for epoch in range(epochs):

            train_loss = self.forward(train_loader, is_training=True, name="Train Loss", verbose=False)
            test_loss = self.forward(test_dataloader, is_training=False, name="Test Loss", verbose=False)

            self.losses['train'].append(train_loss)
            self.losses['test'].append(test_loss)

            if test_loss < self.best_loss:
                self.best_loss = test_loss
                self.save_best_model()
            if verbose and epoch % 10 == 0:
                print(
                    f"Epoch : {epoch + 1}/{epochs} ; LR : {self.optimizer.param_groups[0]['lr']} ; " +
                    f"Train Loss : {train_loss} ; Test Loss : {test_loss} ; Best Loss : {self.best_loss}"
                )
            if self.scheduler is not None:
                self.scheduler.step()

        return self.losses

    @staticmethod
    def compute_metrics(y_trues: Tensor, y_preds: Tensor, num_classes: int = 1) -> ModelMetrics:
        """
        This method of ModelTrainer class is responsible for computing the metrics

        Args:
            y_trues: Tensor. True labels
            y_preds: Tensor. Predicted labels
            num_classes: int. Number of classes. Default is 1.
        """
        try:
            accuracy = Accuracy(task="multiclass", num_classes=num_classes, average='macro')
            precision = Precision(task="multiclass", num_classes=num_classes, average='macro')
            recall = Recall(task="multiclass", num_classes=num_classes, average='macro')
            f1 = F1Score(task="multiclass", num_classes=num_classes, average='macro')

            accuracy_score = accuracy(y_trues, y_preds)
            precision_score = precision(y_trues, y_preds)
            recall_score = recall(y_trues, y_preds)
            f1_score = f1(y_trues, y_preds)

            return ModelMetrics(
                accuracy=accuracy_score.item(),
                precision=precision_score.item(),
                recall=recall_score.item(),
                f1_score=f1_score.item()
            )
        except Exception as e:
            raise APException(e, sys)

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
            y_trues = { # noqa
                "action": torch.empty((0, self.name_vocab_size['action']), device=self.device),
                "duration": torch.empty((0, self.name_vocab_size['duration']), device=self.device),
                "distance": torch.empty((0, self.name_vocab_size['distance']), device=self.device)
            }
            y_preds = { # noqa
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
                y_trues["action"] = torch.cat((y_trues["action"], y1), dim=0) # noqa
                y_trues["duration"] = torch.cat((y_trues["duration"], y2), dim=0)
                y_trues["distance"] = torch.cat((y_trues["distance"], y3), dim=0)

                y_preds["action"] = torch.cat((y_preds["action"], y1_pred), dim=0) # noqa
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
                + pn.aes(x='epoch', y='loss', color='type') # noqa
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

    def generate(self, X, temperature=1.0, do_sample=False, top_k=None) -> np.ndarray:
        with torch.no_grad():
            idx = self.action_start_idx
            while idx < self.max_sequence_length:
                y1, y2, y3 = self.model(torch.tensor(X).to(self.device), training=False)

                y1_probs = torch.softmax(y1, dim=1) / temperature
                y2_probs = torch.softmax(y2, dim=1) / temperature
                y3_probs = torch.softmax(y3, dim=1) / temperature

                if top_k is not None:
                    top_y1 = torch.topk(y1_probs, top_k, dim=1)
                    top_y2 = torch.topk(y2_probs, top_k, dim=1)
                    top_y3 = torch.topk(y3_probs, top_k, dim=1)

                    # Replace the probabilities with the top k probabilities and set the rest to -inf
                    y1_probs = torch.full_like(y1_probs, -float('inf'))
                    y2_probs = torch.full_like(y2_probs, -float('inf'))
                    y3_probs = torch.full_like(y3_probs, -float('inf'))

                    y1_probs.scatter_(1, top_y1.indices, top_y1.values)
                    y2_probs.scatter_(1, top_y2.indices, top_y2.values)
                    y3_probs.scatter_(1, top_y3.indices, top_y3.values)

                if do_sample:
                    y1 = torch.multinomial(y1_probs, num_samples=1)
                    y2 = torch.multinomial(y2_probs, num_samples=1)
                    y3 = torch.multinomial(y3_probs, num_samples=1)
                else:
                    y1 = torch.argmax(y1_probs, dim=1)
                    y2 = torch.argmax(y2_probs, dim=1)
                    y3 = torch.argmax(y3_probs, dim=1)

                y1 = self.tokenizer.convert_index_from_name("action", y1.cpu().numpy())
                y2 = self.tokenizer.convert_index_from_name("duration", y2.cpu().numpy())
                y3 = self.tokenizer.convert_index_from_name("distance", y3.cpu().numpy())

                X[:, idx:(idx + 3)] = np.stack((y1, y2, y3), axis=1) # noqa
                idx += 3
        return X

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
