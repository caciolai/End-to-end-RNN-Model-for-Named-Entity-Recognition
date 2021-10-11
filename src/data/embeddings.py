from typing import *
from tqdm import tqdm

import os
import torch
import numpy as np

import urllib
import zipfile

from .vocabularies import Vocabulary

data_folder_path = "../../data"
model_folder_path = "../../model"
embeddings_folder = os.path.join(model_folder_path, "embeddings")


def create_embedding_tensor(vocabulary: Vocabulary, embedding_dim: int,
                            pretrained_embeddings: Dict[str, np.ndarray]) -> torch.Tensor:
    """
    Creates a lookup tensor for the tokens in the vocabulary starting from pretrained embeddings.

    Args:
        vocabulary:             The vocabulary with the mapping from tokens to indices.
        embedding_dim:          The dimension of the vectors of the embeddings.
        pretrained_embeddings:  The pretrained embeddings for the tokens.
    
    Returns:
        The lookup tensor of shape (vocabulary length, embedding dimension) 
        with the available pretrained embeddings for the tokens in the vocabulary
    """
    embedding_tensor = torch.randn(len(vocabulary), embedding_dim)
    initialised = 0
    for i, w in vocabulary.itos.items():
        if w not in pretrained_embeddings:
            # check needed for <pad>, <unk> tokens
            continue
        
        initialised += 1
        vec = pretrained_embeddings[w]
        embedding_tensor[i] = torch.from_numpy(vec)         
        
    embedding_tensor[vocabulary["<pad>"]] = torch.zeros(embedding_dim)
    print("Initialised embeddings {}".format(initialised))
    print("Random initialised embeddings {} ".format(len(vocabulary) - initialised))
    return embedding_tensor


def get_embeddings(emb_fpath: str, vocab: Vocabulary, emb_dim: int) -> torch.Tensor:
    emb_dict = dict()
    with open(emb_fpath, "r") as f:
        for l in tqdm(f, desc="Loading pretrained word embeddings"):
            line = l.split()
            if len(line) == 2:
                # fasttext has an header to be skipped
                continue
            tok = "".join(line[:-emb_dim])
            if tok in vocab.stoi.keys():
                vec = np.array(line[-emb_dim:], dtype=np.float32)
                emb_dict[tok] = vec
    
    return create_embedding_tensor(vocab, emb_dim, emb_dict)


def download_embeddings(embeddings_fpath: str):
    url = 'https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip'
    filehandle, _ = urllib.request.urlretrieve(url)
    zip_file_object = zipfile.ZipFile(filehandle, 'r')
    first_file = zip_file_object.namelist()[0]
    file = zip_file_object.open(first_file)
    content = file.read()

    with open(embeddings_fpath, "w+") as f:
        f.write(content)

    file.close()


def prepare_embeddings(vocabulary: Vocabulary) -> torch.Tensor:

    embeddings_fname = "crawl-300d-2M.vec"

    embeddings_fpath = os.path.join(embeddings_folder, embeddings_fname)
    embedding_dim = 300

    if not os.path.isfile(embeddings_fpath):
        print("Downloading pre-trained word embeddings...")
        download_embeddings(embeddings_fpath)
        print("Done!")

    pretrained_embeddings = get_embeddings(embeddings_fpath, vocabulary, embedding_dim)

    return pretrained_embeddings

