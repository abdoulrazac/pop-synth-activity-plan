import sys

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import plotnine as pn

from typing import Dict, Tuple

from torch import Tensor

from ap_gpt.components.data_tokenizer import DataTokenizer
from ap_gpt.entity.artifact_entity import DataToSequenceArtifact, ModelTrainerArtifact
from ap_gpt.entity.config_entity import ModelConfig, ModelTrainerConfig
from ap_gpt.models.ap_model_base import APBaseModel
from ap_gpt.utils.data_loader import DataLoader
from ap_gpt.ap_exception import APException
from ap_gpt.ap_logger import logging
from ap_gpt.utils.main_utils import read_data


class ModelTrainer:
    def __init__(
            self,
            model : APBaseModel,
            model_config : ModelConfig,
            tokenizer : DataTokenizer,
            data_to_sequence_artifact: DataToSequenceArtifact,
            model_trainer_config : ModelTrainerConfig = ModelTrainerConfig(),
    ):
        self.model_config = model_config
        self.data_to_sequence_artifact = data_to_sequence_artifact
        self.model_trainer_config = model_trainer_config
        self.tokenizer = tokenizer


        self.model = model(model_config).to(model_config.device)

        self.device = model_config.device
        self.name_vocab_size = model_config.name_vocab_size
        self.pad_token_idx = model_config.pad_token_idx
        self.max_len = model_config.max_len
        self.action_start_idx = model_config.action_start_idx

        self.best_model_path = model_trainer_config.best_model_path

        # Optimization
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        self.criterion = nn.CrossEntropyLoss().to(model_config.device)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=40, gamma=0.1)

        self.losses = {'train': [], 'test': []}
        self.best_loss = float('inf')

    def create_data_loader(self, X : np.ndarray, y:np.ndarray, shuffle=True) -> DataLoader:
        """
        This method of ModelTrainer class is responsible for creating data loader
        """
        try:
            logging.info("Create data loader")
            return DataLoader(X, y, batch_size=self.model_config.batch_size, shuffle=shuffle)
        except Exception as e:
            raise APException(e, sys)

    def value_to_vector(self, value) -> Tuple[Tensor, Tensor, Tensor]:
        y1 = torch.eye(self.name_vocab_size['action']).to(self.device)[value[:, 0]]
        y2 = torch.eye(self.name_vocab_size['duration']).to(self.device)[value[:, 1]]
        y3 = torch.eye(self.name_vocab_size['distance']).to(self.device)[value[:, 2]]

        return y1, y2, y3

    def save_best_model(self) -> None:
        torch.save(self.model.state_dict(), self.best_model_path)

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

            X = X.to(self.device)
            y = y.to(self.device)

            if is_training:
                self.optimizer.zero_grad()

            y1, y2, y3 = self.value_to_vector(y)
            y_hat = self.model(X)
            loss = self.criterion(y_hat[0], y1) + self.criterion(y_hat[1], y2) + self.criterion(y_hat[2], y3)

            if is_training:
                loss.backward()
                self.optimizer.step()

            losses.append(loss.item())

        loss = torch.mean(torch.tensor(losses))
        if verbose:
            print(f"{name} : {np.mean(loss)}")
        return torch.mean(loss)

    def train(self, train_loader: DataLoader, test_dataloader : DataLoader, epochs=10) -> Dict[str, list]:
        for epoch in range(epochs):

            train_loss = self.forward(train_loader, is_training=True, name="Train Loss", verbose=False)
            test_loss = self.forward(test_dataloader, is_training=False, name="Test Loss", verbose=False)

            self.losses['train'].append(train_loss)
            self.losses['test'].append(test_loss)

            if test_loss < self.best_loss:
                self.best_loss = test_loss
                self.save_best_model()

            print(
                f"Epoch : {epoch + 1}/{epochs} ; LR : {self.optimizer.param_groups[0]['lr']} ; Train Loss : {train_loss} ; Test Loss : {test_loss} ; Best Loss : {self.best_loss}")
            if self.scheduler is not None:
                self.scheduler.step()

        return self.losses

    def evaluate(self, test_loader: DataLoader):
        """
        This method of ModelTrainer class is responsible for evaluating the model
        """
        try:
            logging.info("Evaluating the model")
            test_loss = self.forward(test_loader, is_training=False, name="Test Loss", verbose=True)
            return test_loss
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
                + pn.aes(x='epoch', y='loss', color='type')
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

    def generate(self, X) -> np.ndarray:
        with torch.no_grad():
            idx = self.action_start_idx
            while idx < self.max_len:
                y = self.model(torch.tensor(X).to(self.device))
                y1, y2, y3 = torch.argmax(y[0], dim=1), torch.argmax(y[1], dim=1), torch.argmax(y[2], dim=1)
                y1 = self.tokenizer.convert_index_from_name("action", y1.cpu().numpy())
                y2 = self.tokenizer.convert_index_from_name("duration", y2.cpu().numpy())
                y3 = self.tokenizer.convert_index_from_name("distance", y3.cpu().numpy())

                X[:, idx:(idx + 3)] = np.stack((y1, y2, y3), axis=1)
                idx += 3
        return X

    def initiate_training(self) -> ModelTrainerArtifact:
        """
        This method of ModelTrainer class is responsible for initiating the training
        """
        try:
            logging.info("Load datasets")
            x_train = read_data(self.data_to_sequence_artifact.train_x_data_as_sequence_file_path, is_array=True)
            y_train = read_data(self.data_to_sequence_artifact.train_y_data_as_sequence_file_path, is_array=True)
            x_test = read_data(self.data_to_sequence_artifact.test_x_data_as_sequence_file_path, is_array=True)
            y_test = read_data(self.data_to_sequence_artifact.test_y_data_as_sequence_file_path, is_array=True)

            logging.info("Create data loaders")
            train_loader = self.create_data_loader(x_train, y_train, shuffle=True)
            test_loader = self.create_data_loader(x_test, y_test, shuffle=False)

            logging.info("Start training")
            self.train(train_loader, test_loader, epochs=self.model_config.epochs)

            logging.info("Evaluate the model")
            test_loss = self.evaluate(test_loader)

            model_trainer_artifact = ModelTrainerArtifact()

            return model_trainer_artifact
        except Exception as e:
            raise APException(e, sys)
