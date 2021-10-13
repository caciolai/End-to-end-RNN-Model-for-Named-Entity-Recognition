from typing import *

import os
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .datasets import *
from .embeddings import prepare_embeddings

data_folder_path = "../../data"
model_folder_path = "../../model"

train_dataset_path = os.path.join(data_folder_path, "train.tsv")
dev_dataset_path = os.path.join(data_folder_path, "dev.tsv")
test_dataset_path = os.path.join(data_folder_path, "test.tsv")


hparams = {
    "window_size": 25,
    "char_encoding_dim": 10,      # 5 for prefix, 5 for suffix --> 10
    "min_freq_vocab": 1,          # considering all training tokens
    "min_freq_char": 1,           # considering all characters
    "hidden_dim": 128,
    "char_embedding_dim": 50,
    "pos_embedding_dim": 10,
    "bidirectional": True,
    "num_layers": 2,
    "char_num_layers": 1,
    "dropout": 0.5,
    "batch_size": 128,
    "lr": 1e-3,
    "l2_reg":0
}


def prepare_datasets() -> Tuple[Dataset, Dataset, Dataset]:
    train_tokens, train_labels = read_dataset(train_dataset_path)
    dev_tokens, dev_labels = read_dataset(dev_dataset_path)
    test_tokens, test_labels = read_dataset(test_dataset_path)

    # building datasets
    train_dataset = CustomDatasetPOS(train_tokens, train_labels, hparams["window_size"])
    dev_dataset = CustomDatasetPOS(dev_tokens, dev_labels, hparams["window_size"])

    # building vocabularies from training dataset
    vocabulary = build_vocab(train_dataset, min_freq=hparams["min_freq_vocab"])
    label_vocabulary = build_label_vocab(train_dataset)
    char_vocabulary = build_char_vocab(train_dataset, min_freq=hparams["min_freq_char"])
    pos_vocabulary = build_pos_vocab(train_dataset)

    pretrained_embeddings = prepare_embeddings()
    
    hparams.update({
        "vocab_size": len(vocabulary),
        "char_vocab_size": len(char_vocabulary),
        "pos_vocab_size": len(pos_vocabulary),
        "num_classes": len(label_vocabulary),
        "embeddings": pretrained_embeddings
    })

    test_dataset = CustomDatasetPOS(test_tokens, test_labels, hparams["window_size"])
    test_dataset.index_dataset(vocabulary, label_vocabulary, pos_vocabulary,
                            char_vocabulary, char_encoding_dim=hparams["char_encoding_dim"])

    ## indexing datasets
    # word embeddings
    # train_dataset.index_dataset(vocabulary, label_vocabulary,
    #                             char_encoding_dim=char_encoding_dim)
    # dev_dataset.index_dataset(vocabulary, label_vocabulary,
    #                           char_encoding_dim=char_encoding_dim)

    # # word, char embeddings
    # train_dataset.index_dataset(vocabulary, label_vocabulary,
    #                             char_vocabulary, char_encoding_dim=char_encoding_dim)
    # dev_dataset.index_dataset(vocabulary, label_vocabulary,
    #                             char_vocabulary, char_encoding_dim=char_encoding_dim)

    # word, char, pos embeddings
    train_dataset.index_dataset(vocabulary, label_vocabulary, pos_vocabulary,
                                char_vocabulary, char_encoding_dim=hparams["char_encoding_dim"])

    dev_dataset.index_dataset(vocabulary, label_vocabulary, pos_vocabulary,
                                char_vocabulary, char_encoding_dim=hparams["char_encoding_dim"])

    test_dataset.index_dataset(vocabulary, label_vocabulary, pos_vocabulary,
                                char_vocabulary, char_encoding_dim=hparams["char_encoding_dim"])

    return train_dataset, dev_dataset, test_dataset


def prepare_dataloaders() -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_dataset, dev_dataset, test_dataset = prepare_datasets()

    train_dataloader = DataLoader(train_dataset, batch_size=hparams["batch_size"], shuffle=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=hparams["batch_size"], shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=hparams["batch_size"])

    return train_dataloader, dev_dataloader, test_dataloader

