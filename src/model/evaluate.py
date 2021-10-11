from typing import *

from tqdm import tqdm
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from ..model.train import DEVICE
from ..data.vocabularies import *


suffix = "03-05_CharPOSLSTM"

fname_model = "model_params_{}.pt".format(suffix)
fname_hparams = "hparams_{}.json".format(suffix)
fname_vocab = "vocab_{}.json".format(suffix)
fname_vocab_labels = "vocab_labels_{}.json".format(suffix)
fname_char_vocab = "vocab_char_{}.json".format(suffix)
fname_pos_vocab = "vocab_pos_{}.json".format(suffix)

model_folder_path = "../../model"

def compute_predictions(
    model: nn.Module, 
    device: str, 
    dataloader: DataLoader, 
    label_vocab: Vocabulary,
    include_chars: bool=False, 
    include_pos: bool=False) -> Tuple[list, list]:
    """
    Computes and returns all the true and predicted labels of a model on the dataset.

    Args:
        model:          The model to compute the predictions.
        device:         CPU or CUDA.
        dataloader:     The dataset to classify.
        label_vocab:    The vocabulary for the labels.
    
    Returns:
        The true and predicted labels for the data in dataset.
    """
    all_y_pred = list()
    all_y_true = list()
    model.eval()
    with torch.no_grad():
        for sample in tqdm(dataloader, desc="Computing predictions", total=len(dataloader)):
            x = sample['tokens'].to(device)
            y_true = sample['labels']

            x_args = [x]
            if include_chars:
                x_chars = sample["chars"].to(device)
                x_args.append(x_chars)
            
            if include_pos:
                x_pos = sample["pos_tags"].to(device)
                x_args.append(x_pos)

            y_pred = torch.argmax(model(*x_args), -1).view(-1)
            y_true = y_true.view(-1)
            
            valid_indices = y_true != label_vocab['<pad>']
            valid_predictions = y_pred[valid_indices]
            valid_labels = y_true[valid_indices]
            
            all_y_true.extend(valid_labels.tolist())
            all_y_pred.extend(valid_predictions.tolist())
    
    return np.array(all_y_true), np.array(all_y_pred)


def compute_classification_report(y_true:List[int], y_pred:List[int], 
                                  label_vocab:Vocabulary):
    """
    Computes and prints the classification report of the predictions of a model.

    Args:
        y_true:         Ground truth labels.
        y_pred:         Predicted labels.
        label_vocab:    Vocabulary with the labels for decoding.
    """
    print(classification_report(
        y_true, y_pred, zero_division=0, digits=4,
        labels=[i for i,l in label_vocab.itos.items() if l != "<pad>"],
        target_names=[l for i,l in label_vocab.itos.items() if l != "<pad>"]
    ))


def plot_confusion_matrix(y_true: List[int], y_pred: List[int], 
                          label_vocab: Vocabulary, 
                          font_size: Optional[int] = 16,
                          normalize: Optional[bool] = False, 
                          title: Optional[str] = None, 
                          cmap: Optional[str] = plt.cm.Blues):
    """
    Plots the confusion matrix of the predictions of a model.

    Args:
        y_true:         Ground truth labels.
        y_pred:         Predicted labels.
        label_vocab:    Vocabulary with the labels for decoding.
        font_size:      Font size.
        title:          Title of the figure.
        cmap:           Colormap to use for visualization.
    """
    
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    labels = [i for i,l in label_vocab.itos.items() if l != "<pad>"]
    target_names = [l for i,l in label_vocab.itos.items() if l != "<pad>"]
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, None]

    fig, ax = plt.subplots(figsize=(8,8))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    
    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.5 inch.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.5)

    fig.colorbar(im, cax=cax)

    # Set ticks and labels
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=target_names,
           yticklabels=target_names)
    
    ax.tick_params(labelsize=font_size)
    cax.tick_params(labelsize=font_size)           
    ax.set_ylabel("True label", fontdict={"fontsize": font_size})
    ax.set_xlabel("Predicted label", fontdict={"fontsize": font_size})
    
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data to create text annotations.
    fmt = '.4f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    size=font_size)
    
    ax.set_title(title, fontdict={"fontsize": font_size})
    fig.tight_layout()
    plt.grid(False)
    plt.show()


def test(
    model: nn.Module,
    test_dataloader: DataLoader,
    label_vocabulary: LabelVocabulary,
    device: str = DEVICE,
    include_char = True,
    include_pos = True
    ):
    
    test_true, test_pred = compute_predictions(
        model, device, test_dataloader, label_vocabulary, 
        include_chars=include_char, include_pos=include_pos
    )

    compute_classification_report(test_true, test_pred, label_vocabulary)

    plot_confusion_matrix(test_true, test_pred, label_vocabulary, normalize=True)
