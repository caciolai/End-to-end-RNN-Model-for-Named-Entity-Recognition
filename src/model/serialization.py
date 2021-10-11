from typing import *

import os
import torch
import torch.nn as nn

from ..model.train import Trainer
from ..data.vocabularies import *
from ..data.utils import json_dump, json_load
from ..model.models import CharPOSEmbeddingLSTMModel


DEVICE = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )

suffix = "03-05_CharPOSLSTM"

fname_model = "model_params_{}.pt".format(suffix)
fname_hparams = "hparams_{}.json".format(suffix)
fname_vocab = "vocab_{}.json".format(suffix)
fname_vocab_labels = "vocab_labels_{}.json".format(suffix)
fname_char_vocab = "vocab_char_{}.json".format(suffix)
fname_pos_vocab = "vocab_pos_{}.json".format(suffix)

data_folder_path = "../../data"
model_folder_path = "../../model"
weights_folder = os.path.join(model_folder_path, "weights")


def save_model_parameters(model, path, fname):
    fpath = os.path.join(path, fname)
    torch.save(model.state_dict(), fpath)

def load_model_parameters(model, fpath, device=DEVICE):
    model.load_state_dict(torch.load(fpath, map_location=torch.device(device)))

def save_training(
    trainer : Trainer, 
    hyperparams : dict,
    vocabulary : TokenVocabulary,
    label_vocabulary : LabelVocabulary,
    char_vocabulary : CharVocabulary,
    pos_vocabulary : POSVocabulary
    ):


    save_model_parameters(trainer.model, weights_folder, fname_model)
    json_dump(hyperparams, weights_folder, fname_hparams)
    vocabulary.save(weights_folder, fname_vocab)
    label_vocabulary.save(weights_folder, fname_vocab_labels)
    char_vocabulary.save(weights_folder, fname_char_vocab)
    pos_vocabulary.save(weights_folder, fname_pos_vocab)

def load_training() -> Tuple[nn.Module, List[Vocabulary]]:

    hparams = json_load(os.path.join(weights_folder, fname_hparams))
    # model = BasicLSTMModel(hparams)
    # model = CharEmbeddingLSTMModel(hparams)
    model = CharPOSEmbeddingLSTMModel(hparams)
    model.to(DEVICE)
    load_model_parameters(model, os.path.join(weights_folder, fname_model))

    vocabulary = TokenVocabulary()
    label_vocabulary = LabelVocabulary()
    char_vocabulary = CharVocabulary()
    pos_vocabulary = POSVocabulary()
    vocabulary.load(os.path.join(weights_folder, fname_vocab))
    label_vocabulary.load(os.path.join(weights_folder, fname_vocab_labels))
    char_vocabulary.load(os.path.join(weights_folder, fname_char_vocab))
    pos_vocabulary.load(os.path.join(weights_folder, fname_pos_vocab))

    vocabularies = [vocabulary, label_vocabulary, char_vocabulary, pos_vocabulary]

    return model, vocabularies

