
from typing import *

from collections import Counter
import torch

from .utils import json_dump, json_load


class Vocabulary():
    def __init__(self, counter: Counter=None, specials: List[str]=[], min_freq: int = 1):
        """
        Custom base vocabulary class. 

        Args:
            counter:    Counter object with token occurrences.
            specials:   List of special tokens.
            min_freq:   Minimum number of occurrences for a token to be considered in
                        the vocabulary. (default: 1, all the tokens are considered)
        
        Attributes:
            stoi (dict):    Mapping from token strings (s) to indices (i)
            itos (dict):    Mapping from indices (i) to token strings (s)
        """

        self.stoi = dict()
        self.itos = dict()
        
        if counter is not None:
            idx = 0
            for s in specials:
                self.stoi[s] = idx
                self.itos[idx] = s
                idx += 1

            for tok, freq in counter.items():
                if freq >= min_freq:
                    self.stoi[tok] = idx
                    self.itos[idx] = tok
                    idx += 1
    
    def save(self, path: str, fname: str):
        """
        Saves the vocabulary (its two dictionaries) on disk.

        Args:
            path:   Path of the folder to save into.
            fname:  Name of the file to save onto.
        """
        state = {"stoi": self.stoi, "itos": self.itos}
        json_dump(state, path, fname)
    
    def load(self, fpath):
        """
        Loads the vocabulary (its two dictionaries) from disk.
        
        Args:
            fpath:  Path of the file to load from.
        """
        state = json_load(fpath)
        self.stoi = {s: int(i) for s, i in state["stoi"].items()}
        self.itos = {int(i): s for i, s in state["itos"].items()}

    def __getitem__(self, key):
        if type(key) == str:
            return self.stoi[key]
        elif type(key) == int:
            return self.itos[key]
        else:
            raise KeyError("{} is neither a token nor an index".format(key))
    
    def __len__(self):
        return len(self.itos)
    
    def __contains__(self, key):
        if type(key) == str:
            return key in self.stoi
        elif type(key) == int:
            return key in self.itos
        else:
            raise KeyError("{} is neither a token nor an index".format(key))


class TokenVocabulary(Vocabulary):
    def __init__(self, counter: Counter=None, specials: List[str]=[], min_freq: int = 1):
        """
        Custom vocabulary class for tokens. 
        """
        super().__init__(counter, specials, min_freq)

    def encode_tokens(self, sentence: List[dict]) -> List[int]:
        """
        Args:
            sentence: List of dictionaries with keys ("token", "label")
        Returns:
            The list of indices corresponding to the input tokens.
        """
        indices = list()
        for elem in sentence:
            if elem is None:
                indices.append(self["<pad>"])
            elif elem["token"] in self.stoi: # vocabulary string to integer
                indices.append(self[elem["token"]])
            else:
                indices.append(self["<unk>"])
        return indices


class LabelVocabulary(Vocabulary):
    def __init__(self, counter: Counter=None, specials: List[str]=[], min_freq: int = 1):
        """
        Custom vocabulary class for labels.
        """
        super().__init__(counter, specials, min_freq)
    
    def encode_labels(self, sentence: List[dict]) -> List[int]:
        indices = list()
        for elem in sentence:
            if elem is None:
                indices.append(self["<pad>"])
            elif elem["label"] in self.stoi: # vocabulary string to integer
                indices.append(self[elem["label"]])
            else:
                indices.append(self["<unk>"])
        
        return indices
    
    def decode_output(self, outputs:torch.Tensor) -> List[List[str]]:
        """
        Decodes the output of a model that has been trained on a dataset whose labels 
        have been indexed by this vocabulary.
        Args:
            outputs (Tensor):  Tensor with shape (batch_size, max_len, label_vocab_size)
                                containing the predictions of a model .
        Returns:
            The list of dimension (batch_size, max_len) with the decoded labels
        """
        batch_max_indices = torch.argmax(outputs, -1).tolist() # shape = (batch_size, max_len)
        predictions = list()
        for sample_max_indices in batch_max_indices:
            # vocabulary integer to string is used to obtain the corresponding word from the max index
            predictions.append([self.itos[i] for i in sample_max_indices])
        return predictions


class CharVocabulary(Vocabulary):
    def __init__(self, counter: Counter=None, specials: List[str]=[], min_freq: int = 1):
        """
        Custom vocabulary class for characters.
        """
        super().__init__(counter, specials, min_freq)
    
    def encode_chars(self, sentence: List[dict], k:int=5) -> List[List[int]]:
        """
        Computes a character-level encoding of the tokens in the sentence.
        To ensure homogeneity, I pick the first k chars and last k chars 
        of the token (to reflect prefix and suffix), padding in token is shorter
        Args:
            sentence:   List of dictionaries with keys ("token", "label")
            k:          Dimension of the prefix and suffix window of the word to consider.
        Returns:
            The list of list of indices corresponding to the characters of the input tokens.
        """
        sent_indices = list()
        for elem in sentence:
            if elem is None:
                # padding
                tok_indices = [self["<pad>"] for _ in range(2*k)]
            else:
                pre_indices = list()
                suf_indices = list()
                tok_indices = list()
                tok = elem["token"]
                for idx in range(k):
                    if idx >= len(tok):
                        # if token is shorter than k, fill with padding
                        pre_indices.append(self["<pad>"])
                        suf_indices.insert(0, self["<pad>"])
                    else:
                        pre_char = elem["token"][idx]
                        suf_char = elem["token"][-(idx+1)]
                        # prefix char
                        if pre_char in self.stoi: 
                            pre_indices.append(self[pre_char])
                        else:
                            pre_indices.append(self["<unk>"])
                        # suffix char
                        if suf_char in self.stoi: 
                            suf_indices.insert(0, self[suf_char])
                        else:
                            suf_indices.insert(0, self["<unk>"])
                
                # concatenate prefix indices and suffix indices in a 2k
                # character level representation of the token
                tok_indices.extend(pre_indices) 
                tok_indices.extend(suf_indices)
            
            sent_indices.append(tok_indices)
    
        return sent_indices


class POSVocabulary(Vocabulary):
    def __init__(self, counter: Counter=None, specials: List[str]=[], min_freq: int = 1):
        """
        Custom vocabulary class for tokens. 
        """
        super().__init__(counter, specials, min_freq)

    def encode_tags(self, sentence: List[dict]) -> List[int]:
        """
        Args:
            sentence: List of dictionaries with keys ("token", "label", "pos")
        Returns:
            The list of indices corresponding to the input POS tags.
        """
        indices = list()
        for elem in sentence:
            if elem is None:
                indices.append(self["<pad>"])
            elif elem["token"] in self.stoi: # vocabulary string to integer
                indices.append(self[elem["pos"]])
            else:
                indices.append(self["<unk>"])
        return indices

