from typing import *
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
import nltk
from nltk import pos_tag

from .vocabularies import TokenVocabulary, LabelVocabulary, CharVocabulary, POSVocabulary


def read_dataset(path: str) -> Tuple[List[List[str]], List[List[str]]]:
    """
    Reads a dataset from a given path.

    Args:
        path: Path of the file stored in tsv format.

    Returns:
        A 2D list of tokens and another of associated labels.
    """
    tokens_s = []
    labels_s = []

    tokens = []
    labels = []

    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('# '):
                tokens = []
                labels = []
            elif line == '':
                tokens_s.append(tokens)
                labels_s.append(labels)
            else:
                _, token, label = line.split('\t')
                tokens.append(token)
                labels.append(label)

    assert len(tokens_s) == len(labels_s)

    return tokens_s, labels_s

class CustomDataset(Dataset):
    def __init__(self, sentences: List[List[str]], labels: List[List[str]], 
                 window_size:int, window_shift:int=-1):
        """
        Custom dataset class for tokens and labels.
        
        Args:
            sentences:      The tokens organized in sentences.
            labels:         The labels of the tokens.
            window_size:    The maximum length of a sentence in terms of 
                            number of tokens.
            window_shift:   The number of tokens we shift the window 
                            over the sentence. 
                            Default value is -1 meaning that the window will
                            be shifted by window_size.
        
        Attributes:
            data:           (list(dict("inputs", "outputs", "chars"*))).
            encoded_data:   List of indices from data, given a vocabulary.
        """

        self.window_size = window_size
        self.window_shift = window_shift if window_shift > 0 else window_size
        self.data = self.create_windows(sentences, labels)
        self.encoded_data = None
    
    def index_dataset(self, 
                      vocabulary: TokenVocabulary, label_vocabulary: LabelVocabulary, 
                      char_vocabulary: Optional[CharVocabulary] = None, 
                      char_encoding_dim: int = 10):
        """
        Indexes the data using the correspondences given by vocabularies.

        Args:
            vocabulary:         Vocabulary for the tokens.
            label_vocabulary:   Vocabulary for the labels.
            char_vocabulary:    Character-level vocabulary.
            char_encoding_dim:  Dimension of the window to use to derive 
                                character-level representation of the tokens
        """
        
        self.encoded_data = list()
        for window in tqdm(self.data, desc="Indexing dataset"):
            encoded_tokens = torch.LongTensor(vocabulary.encode_tokens(window))
            encoded_labels = torch.LongTensor(label_vocabulary.encode_labels(window))
            encoded_elem = {"tokens":encoded_tokens, 
                            "labels":encoded_labels}
            if char_vocabulary is not None:
                encoded_chars = torch.LongTensor(char_vocabulary.encode_chars(window, k=char_encoding_dim//2))
                encoded_elem["chars"] = encoded_chars

            self.encoded_data.append(encoded_elem)

    def create_windows(self, sentences: List[List[str]], labels: List[List[str]]):
        """
        Creates fixed-length windows out of the sentences, to ensure length homogeneity
        across the samples of the dataset, breaking down longer sentences in more windows and
        filling up shorter sentences with padding if necessary.
       
        Args:
            sentences:  List of list of tokens organized in sentences.
            labels:     Labels of the tokens.
        """
        assert len(sentences) == len(labels)

        data = []
        for sentence, sent_labels in tqdm(zip(sentences, labels), desc="Creating windows", total=len(sentences)):
            for i in range(0, len(sentence), self.window_shift):
                tokens = sentence[i:i+self.window_size]
                labels = sent_labels[i:i+self.window_size]
                window = [{"token": t, "label": l} for t, l in zip(tokens, labels)]
                if len(window) < self.window_size:
                    window = window + [None]*(self.window_size - len(window))
                                
                data.append(window)
        
        return data
    
    def get_raw_element(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.encoded_data is None:
            raise RuntimeError("""Trying to retrieve elements but index_dataset
            has not been invoked yet! Be sure to invoce index_dataset on this object
            before trying to retrieve elements. In case you want to retrieve raw
            elements, use the method get_raw_element(idx)""")
        return self.encoded_data[idx]


class CustomDatasetPOS(Dataset):
    def __init__(self, sentences: List[List[str]], labels: List[List[str]], 
                 window_size:int, window_shift:int=-1):
        """
        Custom dataset class for tokens and labels (POS tags).
        
        Args:
            sentences:      The tokens organized in sentences.
            labels:         The labels of the tokens.
            window_size:    The maximum length of a sentence in terms of 
                            number of tokens.
            window_shift:   The number of tokens we shift the window 
                            over the sentence. 
                            Default value is -1 meaning that the window will
                            be shifted by window_size.
        
        Attributes:
            data:           (list(dict("inputs", "outputs", "chars"*))).
            encoded_data:   List of indices from data, given a vocabulary.
        """

        self.window_size = window_size
        self.window_shift = window_shift if window_shift > 0 else window_size
        self.data = self.create_windows(sentences, labels)
        self.encoded_data = None
    
    def index_dataset(self, 
                      vocabulary: TokenVocabulary, label_vocabulary: LabelVocabulary, 
                      pos_vocabulary: POSVocabulary,
                      char_vocabulary: Optional[CharVocabulary] = None, 
                      char_encoding_dim: int = 10):
        """
        Indexes the data using the correspondences given by vocabularies.

        Args:
            vocabulary:         Vocabulary for the tokens.
            label_vocabulary:   Vocabulary for the labels.
            pos_vocabulary:     Vocabulary for the POS tags.
            char_vocabulary:    Character-level vocabulary.
            char_encoding_dim:  Dimension of the window to use to derive 
                                character-level representation of the tokens
        """
        
        self.encoded_data = list()
        for window in tqdm(self.data, desc="Indexing dataset"):
            encoded_tokens = torch.LongTensor(vocabulary.encode_tokens(window))
            encoded_labels = torch.LongTensor(label_vocabulary.encode_labels(window))
            encoded_pos_tags = torch.LongTensor(pos_vocabulary.encode_tags(window))
            encoded_elem = {"tokens":encoded_tokens, 
                            "labels":encoded_labels,
                            "pos_tags": encoded_pos_tags}
            if char_vocabulary is not None:
                encoded_chars = torch.LongTensor(char_vocabulary.encode_chars(window, k=char_encoding_dim//2))
                encoded_elem["chars"] = encoded_chars

            self.encoded_data.append(encoded_elem)

    def create_windows(self, sentences: List[List[str]], labels: List[List[str]]):
        """
        Creates fixed-length windows out of the sentences, to ensure length homogeneity
        across the samples of the dataset, breaking down longer sentences in more windows and
        filling up shorter sentences with padding if necessary.
       
        Args:
            sentences:  List of list of tokens organized in sentences.
            labels:     Labels of the tokens.
        """
        assert len(sentences) == len(labels)

        data = []
        for sentence, sent_labels in tqdm(zip(sentences, labels), desc="Creating windows", total=len(sentences)):
            for i in range(0, len(sentence), self.window_shift):
                tokens = sentence[i:i+self.window_size]
                labels = sent_labels[i:i+self.window_size]
                pos_tags = [elem[1] for elem in pos_tag(tokens)]
                window = [{"token": t, "label": l, "pos": p} for t, l, p in zip(tokens, labels, pos_tags)]
                if len(window) < self.window_size:
                    window = window + [None]*(self.window_size - len(window))

                data.append(window)
        
        return data
    
    def get_raw_element(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.encoded_data is None:
            raise RuntimeError("""Trying to retrieve elements but index_dataset
            has not been invoked yet! Be sure to invoce index_dataset on this object
            before trying to retrieve elements. In case you want to retrieve raw
            elements, use the method get_raw_element(idx)""")
        return self.encoded_data[idx]

def build_vocab(dataset: CustomDataset, min_freq=1) -> TokenVocabulary:
    """
    Builds a vocabulary for the tokens in dataset.

    Args:
        dataset:    The dataset with the considered corpus.
        min_freq:   Minimum number of occurrences for a token in the corpus
                    to be included among the vocabulary tokens
    """
    counter = Counter()
    for i in tqdm(range(len(dataset)), desc="Building vocabulary"):
        for elem in dataset.get_raw_element(i):
            if elem is not None:
                token = elem["token"]
                counter[token]+=1

    # we add special tokens for handling padding and unknown words at testing time.
    return TokenVocabulary(counter, specials=['<pad>', '<unk>'], min_freq=min_freq)

def build_label_vocab(dataset: CustomDataset) -> LabelVocabulary:
    """
    Builds a vocabulary for the labels in dataset.

    Args:
        dataset: The dataset with the considered corpus.
    """
    counter = Counter()
    for i in tqdm(range(len(dataset)), desc="Building label vocabulary"):
        for elem in dataset.get_raw_element(i):
            if elem is not None:
                label = elem["label"]
                counter[label]+=1
    # No <unk> token for labels.
    return LabelVocabulary(counter, specials=['<pad>'])

def build_char_vocab(dataset: CustomDataset, min_freq=1) -> CharVocabulary:
    """
    Builds a vocabulary for the character-level representations of the tokens in dataset.

    Args:
        dataset: The dataset with the considered corpus.
    """
    counter = Counter()
    for i in tqdm(range(len(dataset)), desc="Building character vocabulary"):
        for elem in dataset.get_raw_element(i):
            if elem is not None:
                token = elem["token"]
                for char in token:
                    counter[char]+=1

    # we add special tokens for handling padding and unknown words at testing time.
    return CharVocabulary(counter, specials=['<pad>', '<unk>'], min_freq=min_freq)

def build_pos_vocab(dataset: CustomDataset) -> POSVocabulary:
    counter = Counter()
    for i in tqdm(range(len(dataset)), desc="Building POS vocabulary"):
        for elem in dataset.get_raw_element(i):
            if elem is not None:
                pos = elem["pos"]
                counter[pos]+=1

    # we add special tokens for handling padding and unknown words at testing time.
    return POSVocabulary(counter, specials=['<pad>', '<unk>'])

