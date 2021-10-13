from typing import *

import torch
from torch import nn


class BasicLSTMModel(nn.Module):
    def __init__(self, hparams:Dict[str, Any]):
        """
        A simple (Bi)LSTM model

        Args:
            hparams:        Dictionary with the hyperparameters for the model.

        Attributes:
            word_embedding: Embedding layer to vectorize the input tokens, can be pretrained or not.
            lstm:           LSTM layer with dropout, can be bidirectional or not.
            dropout:        Dropout layer, with same amount of dropout used in lstm.
            linear:         Linear output layer.
        """
        super().__init__()

        self.word_embedding = nn.Embedding(hparams["vocab_size"], hparams["embedding_dim"])
        if hparams["embeddings"] is not None:
            print("Initializing embeddings from pretrained")
            self.word_embedding.weight.data.copy_(hparams["embeddings"])

        self.lstm = nn.LSTM(hparams["embedding_dim"], hparams["hidden_dim"], 
                            bidirectional=hparams["bidirectional"],
                            batch_first=True,
                            num_layers=hparams["num_layers"], 
                            dropout = hparams["dropout"] if hparams["num_layers"] > 1 else 0)
        
        lstm_output_dim = hparams["hidden_dim"] if hparams["bidirectional"] is False \
                                                else hparams["hidden_dim"] * 2
        self.dropout = nn.Dropout(hparams["dropout"])
        self.linear = nn.Linear(lstm_output_dim, hparams["num_classes"])

    
    def forward(self, x):
        # apply embedding
        embeddings = self.word_embedding(x)
        embeddings = self.dropout(embeddings)

        # fed it to lstm
        o, (h, c) = self.lstm(embeddings)
        o = self.dropout(o)
        o = self.linear(o)
        return o


class CharEmbeddingLSTMModel(nn.Module):
    def __init__(self, hparams: Dict[str, Any]):
        """
        A more complex (Bi)LSTM model, that uses character-level embeddings along with
        (pre)trained word embeddings.

        Args:
            hparams:        Dictionary with the hyperparameters for the model.

        Attributes:
            word_embedding: Embedding layer to vectorize the input tokens, can be pretrained or not.
            char_embedding: Embedding layer to vectorize character-level representations.
            char_lstm:      LSTM layer to extract character-level embeddings from the embedded
                            character-level representations of the tokens.
            lstm:           LSTM layer with dropout, can be bidirectional or not.
            dropout:        Dropout layer, with same amount of dropout used in lstm.
            linear:         Linear output layer.
        """
        super().__init__()

        self.word_embedding = nn.Embedding(hparams["vocab_size"], hparams["embedding_dim"])
        if hparams["embeddings"] is not None:
            print("Initializing embeddings from pretrained")
            self.word_embedding.weight.data.copy_(hparams["embeddings"])

        self.char_embedding = nn.Embedding(hparams["char_vocab_size"], hparams["char_embedding_dim"])
        self.char_lstm = nn.LSTM(hparams["char_embedding_dim"], hparams["char_embedding_dim"],
                            bidirectional=hparams["bidirectional"],
                            batch_first=True,
                            num_layers=hparams["char_num_layers"], 
                            dropout = hparams["dropout"] if hparams["char_num_layers"] > 1 else 0)

        lstm_input_dim = hparams["embedding_dim"] + hparams["char_embedding_dim"] \
                            if hparams["bidirectional"] is False \
                             else hparams["embedding_dim"] + hparams["char_embedding_dim"] * 2
        
        self.hidden = nn.Linear(lstm_input_dim, lstm_input_dim)

        self.lstm = nn.LSTM(lstm_input_dim, hparams["hidden_dim"], 
                            bidirectional=hparams["bidirectional"],
                            batch_first=True,
                            num_layers=hparams["num_layers"], 
                            dropout = hparams["dropout"] if hparams["num_layers"] > 1 else 0)
        

        lstm_output_dim = hparams["hidden_dim"] if hparams["bidirectional"] is False \
                                                else hparams["hidden_dim"] * 2

        self.dropout = nn.Dropout(hparams["dropout"])
        self.classifier = nn.Linear(lstm_output_dim, hparams["num_classes"])

    
    def forward(self, x, x_chars):
        """
        Args:
            x:          Batch of dimensions (batch_size, window_size, embedding_dim)
                        with the embedded tokens for each window in the batch
            x_chars:    Batch of dimensions (batch_size, window_size, char_encoding_len, char_embedding_dim)
                        with the embedded characters for each character-level representation
                        for each token for each window in the batch
        """
        # apply embedding
        word_embeddings = self.word_embedding(x)
        word_embeddings = self.dropout(word_embeddings)

        # apply char embedding
        char_embeddings = self.char_embedding(x_chars)
        char_embeddings = self.dropout(char_embeddings)
        
        ### obtaining character-level embeddings (last output of the BiLSTM)
        # reshape to (batch_size * window_size, char_sequence_len, char_embedding_dim)
        in_char_shape = char_embeddings.shape
        char_embeddings = char_embeddings.view(-1, in_char_shape[2], in_char_shape[3])        

        # feed to char lstm
        char_embeddings, (h, c) = self.char_lstm(char_embeddings)
        char_embeddings = self.dropout(char_embeddings)

        # separate forward and backward directions
        forward_ce = char_embeddings[:, :, :in_char_shape[3]]
        backward_ce = char_embeddings[:, :, in_char_shape[3]:]

        # reshape to (batch_size, window_size, char_sequence_len, BiLSTM output dim)
        forward_ce = forward_ce.reshape(in_char_shape[0], in_char_shape[1], in_char_shape[2], -1)
        backward_ce = backward_ce.reshape(in_char_shape[0], in_char_shape[1], in_char_shape[2], -1)
        
        # join back together the last output of both directions
        char_embeddings = torch.cat((forward_ce[:, :, -1, :], backward_ce[:, :, -1, :]), dim=-1)

        # get final embeddings
        embeddings = torch.cat((word_embeddings, char_embeddings), dim=-1)

        # refine embeddings with hidden layer
        embeddings = self.hidden(embeddings)
        self.dropout(embeddings)

        # feed to lstm
        o, (h, c) = self.lstm(embeddings)
        o = self.dropout(o)
        o = self.classifier(o)
        return o


class CharPOSEmbeddingLSTMModel(nn.Module):
    def __init__(self, hparams: Dict[str, Any]):
        """
        A more complex (Bi)LSTM model, that uses character-level embeddings and
        POS tagging embeddings along with (pre)trained word embeddings.

        Args:
            hparams:        Dictionary with the hyperparameters for the model.

        Attributes:
            word_embedding: Embedding layer to vectorize the input tokens, can be pretrained or not.
            char_embedding: Embedding layer to vectorize character-level representations.
            char_lstm:      LSTM layer to extract character-level embeddings from the embedded
                            character-level representations of the tokens.
            pos_lstm:       LSTM layer to extract embeddings from the POS tags of the tokens.
            lstm:           LSTM layer with dropout, can be bidirectional or not.
            dropout:        Dropout layer, with same amount of dropout used in lstm.
            linear:         Linear output layer.
        """
        super().__init__()

        self.word_embedding = nn.Embedding(hparams["vocab_size"], hparams["embedding_dim"])
        if hparams["embeddings"] is not None:
            print("Initializing embeddings from pretrained")
            self.word_embedding.weight.data.copy_(hparams["embeddings"])

        self.char_embedding = nn.Embedding(hparams["char_vocab_size"], hparams["char_embedding_dim"])
        self.char_lstm = nn.LSTM(hparams["char_embedding_dim"], hparams["char_embedding_dim"],
                            bidirectional=hparams["bidirectional"],
                            batch_first=True,
                            num_layers=hparams["char_num_layers"], 
                            dropout = hparams["dropout"] if hparams["char_num_layers"] > 1 else 0)

        self.pos_embedding = nn.Embedding(hparams["pos_vocab_size"], hparams["pos_embedding_dim"])
        
        lstm_input_dim = hparams["embedding_dim"] + hparams["char_embedding_dim"] + hparams["pos_embedding_dim"] \
                             if hparams["bidirectional"] is False \
                             else hparams["embedding_dim"] + hparams["char_embedding_dim"]*2 + hparams["pos_embedding_dim"]
        
        self.hidden = nn.Linear(lstm_input_dim, lstm_input_dim)

        self.lstm = nn.LSTM(lstm_input_dim, hparams["hidden_dim"], 
                            bidirectional=hparams["bidirectional"],
                            batch_first=True,
                            num_layers=hparams["num_layers"], 
                            dropout = hparams["dropout"] if hparams["num_layers"] > 1 else 0)
        

        lstm_output_dim = hparams["hidden_dim"] if hparams["bidirectional"] is False \
                                                else hparams["hidden_dim"] * 2

        self.dropout = nn.Dropout(hparams["dropout"])
        self.classifier = nn.Linear(lstm_output_dim, hparams["num_classes"])

    
    def forward(self, x, x_chars, x_pos):
        """
        Args:
            x:          Batch of dimensions (batch_size, window_size)
                        with the tokens for each window in the batch
            x_chars:    Batch of dimensions (batch_size, window_size, char_encoding_len)
                        with the encoded characters for each character-level representation
                        for each token for each window in the batch
            x_pos:      Batch of dimensions (batch_size, window_size)
                        with the POS tags for the tokens in each window in the batch
        """
        # apply embedding
        word_embeddings = self.word_embedding(x)
        word_embeddings = self.dropout(word_embeddings)
        
        ### obtain character-level embeddings (last output of the BiLSTM)
        # apply char embedding
        char_embeddings = self.char_embedding(x_chars)
        char_embeddings = self.dropout(char_embeddings)
       
        # reshape to (batch_size * window_size, char_sequence_len, char_embedding_dim)
        in_char_shape = char_embeddings.shape
        char_embeddings = char_embeddings.view(-1, in_char_shape[2], in_char_shape[3])        

        # feed to char lstm
        char_embeddings, (h, c) = self.char_lstm(char_embeddings)

        # separate forward and backward directions
        forward_ce = char_embeddings[:, :, :in_char_shape[3]]
        backward_ce = char_embeddings[:, :, in_char_shape[3]:]

        # reshape to (batch_size, window_size, char_sequence_len, BiLSTM output dim)
        forward_ce = forward_ce.reshape(in_char_shape[0], in_char_shape[1], in_char_shape[2], -1)
        backward_ce = backward_ce.reshape(in_char_shape[0], in_char_shape[1], in_char_shape[2], -1)
        
        # join back together the last output of both directions
        char_embeddings = torch.cat((forward_ce[:, :, -1, :], backward_ce[:, :, -1, :]), dim=-1)

        # apply pos embeddings
        pos_embeddings = self.pos_embedding(x_pos)
        pos_embeddings = self.dropout(pos_embeddings)

        # get final embeddings
        embeddings = torch.cat((word_embeddings, char_embeddings, pos_embeddings), dim=-1)

        # refine embeddings with hidden layer
        embeddings = self.hidden(embeddings)
        self.dropout(embeddings)
        
        # feed to lstm
        o, (h, c) = self.lstm(embeddings)
        o = self.dropout(o)
        o = self.classifier(o)
        return o

