import sys
import numpy as np
from typing import List, Tuple, Union, Literal

import pandas as pd

from ap_gpt.ap_exception import APException
from ap_gpt.constants import EOT_TOKEN, PAD_TOKEN, UNK_TOKEN, SOT_TOKEN
from ap_gpt.entity.artifact_entity import DataSplittingArtifact, DataTokenizerArtifact
from ap_gpt.entity.config_entity import DataTokenizerConfig
from ap_gpt.ap_logger import logging
from ap_gpt.utils.main_utils import read_data, save_data


class DataTokenizer:
    """
    Attributes:
        pad_token (str): Padding token.
        unk_token (str): Unknown token.
        sot_token (str): Start of text token.
        eot_token (str): End of text token.
        word2idx (dict): Dictionary mapping words to their indices.
        idx2word (dict): Dictionary mapping indices to their words.
        name2idx (dict): Dictionary mapping names to their indices.
        vocab_size (int): Size of the vocabulary.
        name_vocab_size (dict): Dictionary mapping names to their vocabulary sizes.
    Methods:
        __init__(self, sot_token=SOT_TOKEN, eot_token=EOT_TOKEN, pad_token=PAD_TOKEN, unk_token=UNK_TOKEN):
        _build_vocab(self):
        _add_word(self, word):
        add_words(self, words: Union[List, Tuple, str, np.array, np.ndarray], name: Literal[None, 'action', 'duration', 'distance'] = None):
        _add_word_to(self, name, word):
        encode(self, words: Union[str, List, Tuple, np.array, np.ndarray]):
        decode(self, index: Union[int, List[int], Tuple[int], np.array, np.ndarray]):
        encode_name(self, name, words: Union[str, List, Tuple, np.array, np.ndarray]):
        decode_name(self, name, index: Union[int, List[int], Tuple[int], np.array, np.ndarray]):
        encode_from_name(self, name, words: Union[str, List, Tuple, np.array, np.ndarray]):
        decode_from_name(self, name, index: Union[int, List[int], Tuple[int], np.array, np.ndarray]):
        save(self, path):
        load(self, path):
        __len__(self):
            Returns the size of the vocabulary.
        __repr__(self):
            Returns a string representation of the Tokenizer object.
    """

    def __init__(self,
                 data_splitting_artifact: DataSplittingArtifact,
                 data_tokenizer_config: DataTokenizerConfig = DataTokenizerConfig(),
                 sot_token: str = SOT_TOKEN,
                 eot_token: dict = EOT_TOKEN,
                 pad_token: dict = PAD_TOKEN,
                 unk_token: str  = UNK_TOKEN ):
        """
        Initializes the tokenizer with special tokens and builds the vocabulary.

        Args:
            sot_token (str): Start of text token. Default is SOT_TOKEN.
            eot_token (dict): End of text token. Default is EOT_TOKEN.
            pad_token (dict): Padding token. Default is PAD_TOKEN.
            unk_token (str): Unknown token. Default is UNK_TOKEN.
        """
        self.data_tokenizer_config = data_tokenizer_config
        self.data_splitting_artifact = data_splitting_artifact
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.sot_token = sot_token
        self.eot_token = eot_token
        self.word2idx = {}
        self.idx2word = {}
        self.name2idx = {"action": {}, "duration": {}, "distance": {}}
        self.idx2name = {"action": {}, "duration": {}, "distance": {}}
        self.vocab_size = 0
        self.name_vocab_size = {"action": 0, "duration": 0, "distance": 0}
        self._build_vocab()

    def _build_vocab(self):
        """
        Builds the vocabulary for the tokenizer by adding special tokens.

        This method adds the following tokens to the vocabulary:
        - Padding tokens (pad_token)
        - End of text tokens (eot_token)
        - Unknown token (unk_token)
        - Start of text token (sot_token)

        Each token is added in lowercase.

        Returns:
            None
        """
        for k, v in self.pad_token.items():
            self._add_word_to(k.lower(), v)
        for k, v in self.eot_token.items():
            self._add_word_to(k.lower(), v)
        self._add_word(self.unk_token)
        self._add_word(self.sot_token)

    def _add_word(self, word):
        """
        Adds a word to the vocabulary.

        If the word is not already in the vocabulary, it assigns the word a new index,
        updates the word-to-index and index-to-word mappings, and increments the vocabulary size.

        Args:
            word (str): The word to be added to the vocabulary.

        Returns:
            None
        """
        if word not in self.word2idx:
            self.word2idx[word] = self.vocab_size
            self.idx2word[self.vocab_size] = word
            self.vocab_size += 1

    def add_words(self, words: Union[List, Tuple, str, np.array, np.ndarray],
                  name: Literal[None, 'action', 'duration', 'distance'] = None):
        """
        Adds words to the internal storage and optionally associates them with a category.

        Parameters:
        words (Union[List, Tuple, str, np.array, np.ndarray]): The words to be added. Can be a single string, a list, a tuple, or a numpy array of strings.
        name (Literal[None, 'action', 'duration', 'distance'], optional): The category to associate the words with. Defaults to None.

        Returns:
        None
        """
        if type(words) == str:
            self._add_word(words)
            if name is not None:
                self._add_word_to(name, words)
        elif type(words) == list or type(words) == tuple:
            for word in words:
                self._add_word(word)
                if name is not None:
                    self._add_word_to(name, word)
        elif type(words) == np.array or type(words) == np.ndarray:
            for word in words.flatten():
                self._add_word(word)
                if name is not None:
                    self._add_word_to(name, word)

    def _add_word_to(self, name, word):
        """
        Adds a word to the vocabulary for a given name if it is not already present.

        Args:
            name (str): The name of the vocabulary to which the word should be added.
            word (str): The word to be added to the vocabulary.

        Returns:
            None
        """
        if word not in self.name2idx[name]:
            self.name2idx[name][word] = self.name_vocab_size[name]
            self.idx2name[name][self.name_vocab_size[name]] = word
            self.name_vocab_size[name] += 1

    def encode(self, words: Union[str, List, Tuple, np.array, np.ndarray]):
        """
        Encodes a given input of words into their corresponding indices based on the word2idx dictionary.

        Parameters:
        words (Union[str, List, Tuple, np.array, np.ndarray]): The input words to encode. It can be a single string,
                                                               a list or tuple of strings, or a numpy array of strings.

        Returns:
        Union[int, List[int], np.ndarray]: The encoded indices of the input words. If the input is a single string,
                                           an integer is returned. If the input is a list or tuple, a list of integers
                                           is returned. If the input is a numpy array, a numpy array of integers is returned.
        """
        if type(words) == str:
            return self.word2idx.get(words, self.word2idx[self.unk_token])
        elif type(words) == list or type(words) == tuple:
            return [self.word2idx.get(word, self.word2idx[self.unk_token]) for word in words]
        elif type(words) == np.array or type(words) == np.ndarray:
            out = [self.word2idx.get(word, self.word2idx[self.unk_token]) for word in words.flatten()]
            return np.array(out).reshape(words.shape)

    def decode(self, index: Union[Union[int, List, Tuple, np.array, np.ndarray]]):
        """
        Decodes the given index or indices into corresponding words.

        Parameters:
        index (Union[int, List[int], Tuple[int], np.array, np.ndarray]): The index or indices to decode.
            It can be a single integer, a list of integers, a tuple of integers, or a numpy array of integers.

        Returns:
        Union[str, List[str], np.ndarray]: The decoded word(s).
            If the input is a single integer, a single word (str) is returned.
            If the input is a list or tuple, a list of words (List[str]) is returned.
            If the input is a numpy array, a numpy array of words (np.ndarray) is returned.
        """
        if type(index) == int:
            return self.idx2word.get(index, self.unk_token)
        elif type(index) == list or type(index) == tuple:
            return [self.idx2word.get(i, self.unk_token) for i in index]
        elif type(index) == np.array or type(index) == np.ndarray:
            out = [self.idx2word.get(i, self.unk_token) for i in index.flatten()]
            return np.array(out).reshape(index.shape)

    def convert_index_to_name(self,
                              name: Literal['action', 'duration', 'distance'],
                              index : Union[int, List[int], Tuple[int], np.array, np.ndarray]):
        """
        Converts the index of a word from the general vocabulary to the index of the word in a given name.

        Args:
            name (str): The name of the vocabulary to which the index is converted.
            index (Union[int, List[int], Tuple[int], np.array, np.ndarray]): The index or indices to convert.
                It can be a single integer, a list of integers, a tuple of integers, or a numpy array of integers.

        Returns:
            Union[int, List[int], np.ndarray]: The index or indices of the word(s) in the given vocabulary.
        """
        if type(index) == int:
            return self.name2idx[name].get(self.idx2word.get(index, self.unk_token), self.unk_token)
        elif type(index) == list or type(index) == tuple:
            return [self.name2idx[name].get(self.idx2word.get(i, self.unk_token), self.unk_token) for i in index]
        elif type(index) == np.array or type(index) == np.ndarray:
            out = [self.name2idx[name].get(self.idx2word.get(i, self.unk_token), self.unk_token) for i in
                   index.flatten()]
            return np.array(out).reshape(index.shape)

    def convert_index_from_name(self,
                                name : Literal['action', 'duration', 'distance'],
                                index : Union[int, List[int], Tuple[int], np.array, np.ndarray]):
        """
        Converts the index of a word from a given name to the index of the word in the general vocabulary.

        Args:
            name (str): The name of the vocabulary from which the index is taken.
            index (Union[int, List[int], Tuple[int], np.array, np.ndarray]): The index or indices to convert.
                It can be a single integer, a list of integers, a tuple of integers, or a numpy array of integers.

        Returns:
            Union[int, List[int], np.ndarray]: The index or indices of the word(s) in the general vocabulary.
        """
        if type(index) == int:
            return self.word2idx.get(self.idx2name[name].get(index, self.unk_token), self.unk_token)
        elif type(index) == list or type(index) == tuple:
            return [self.word2idx.get(self.idx2name[name].get(i, self.unk_token), self.unk_token) for i in index]
        elif type(index) == np.array or type(index) == np.ndarray:
            out = [self.word2idx.get(self.idx2name[name].get(i, self.unk_token), self.unk_token) for i in
                   index.flatten()]
            return np.array(out).reshape(index.shape)

    def save(self, path):
        """
        Save the idx2word dictionary to a file.

        Args:
            path (str): The file path where the dictionary will be saved.

        The dictionary is saved in a tab-separated format, with each line containing
        an index and its corresponding word.
        """
        with open(path, 'w') as f:
            for word in self.idx2word.items():
                f.write(f"{word[0]}\t{word[1]}\n")

    def load(self, path):
        """
        Load vocabulary from a file.

        Args:
            path (str): The path to the file containing the vocabulary. The file should have tab-separated values
                        with each line containing an index and a word.

        Raises:
            ValueError: If a line in the file does not contain exactly two tab-separated values.

        Example:
            Given a file with the following content:
            0   hello
            1   world

            The method will populate `self.word2idx` with {'hello': 0, 'world': 1}
            and `self.idx2word` with {0: 'hello', 1: 'world'}.
        """
        with open(path, 'r') as f:
            for line in f.readlines():
                idx, word = line.strip().split("\t")
                self.word2idx[word] = int(idx)
                self.idx2word[int(idx)] = word
            self.vocab_size = len(self.word2idx)

    def initiate_tokenizer(self) -> DataTokenizerArtifact:
        """
        Initiates the tokenizer by loading the vocabulary from a file if is_train is False.
        If is_train is True, it builds the vocabulary from the training data.
        """
        try:
            logging.info("Reading data from the file")
            df_train = read_data(self.data_splitting_artifact.train_data_file_path)
            df_test = read_data(self.data_splitting_artifact.test_data_file_path)

            logging.info("Tokenizing data")
            self.add_words(df_train[[c for c in df_train.columns if not c.startswith(('action', 'duration', 'distance'))]].values)
            self.add_words(df_train["action"].values, name="action")
            self.add_words(df_train["duration"].values, name="duration")
            self.add_words(df_train["distance"].values, name="distance")

            logging.info("Tokenizing data completed")
            df_train_encoded = self.encode(df_train.values)
            df_test_encoded = self.encode(df_test.values)

            logging.info("Saving tokenizer")
            self.save(self.data_tokenizer_config.tokenizer_file_path)

            data_tokenizer_artifact = DataTokenizerArtifact(
                tokenizer_file_path=self.data_tokenizer_config.tokenizer_file_path,
                train_encoded_data_file_path=self.data_tokenizer_config.train_encoded_data_file_path,
                test_encoded_data_file_path=self.data_tokenizer_config.test_encoded_data_file_path,
            )

            logging.info("Save encoded data")
            save_data(df_train_encoded, data_tokenizer_artifact.train_encoded_data_file_path)
            save_data(df_test_encoded, data_tokenizer_artifact.test_encoded_data_file_path)

            logging.info("Tokenizer initiated successfully")
            return data_tokenizer_artifact

        except Exception as e:
            raise APException(e, sys) from e

    def __len__(self):
        return self.vocab_size

    def __repr__(self):
        return f"Tokenizer(vocab_size={self.vocab_size})"
