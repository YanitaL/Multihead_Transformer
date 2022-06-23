import random

import torch
import torch.nn as nn
import torch.optim as optim

class Encoder(nn.Module):
    """ The Encoder module of the Seq2Seq model
    """

    def __init__(self, input_size, emb_size, encoder_hidden_size, decoder_hidden_size, dropout=0.2, model_type="RNN"):
        super(Encoder, self).__init__()

        self.input_size = input_size
        self.emb_size = emb_size
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.model_type = model_type
        self.embedding = nn.Embedding(self.input_size, self.emb_size)
        
        if self.model_type == 'RNN':
            self.rnn = nn.RNN(self.emb_size, self.encoder_hidden_size, batch_first=True, dropout=dropout)
        elif self.model_type == 'LSTM':
            self.rnn = nn.LSTM(self.emb_size, self.encoder_hidden_size, batch_first=True, dropout=dropout)
        
        self.linear1 = nn.Linear(self.encoder_hidden_size, self.encoder_hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(self.encoder_hidden_size, self.decoder_hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        """ The forward pass of the encoder
            Args:
                input (tensor): the encoded sequences of shape (batch_size, seq_len)

            Returns:
                output (tensor): the output of the Encoder;
                hidden (tensor): the weights coming out of the last hidden unit
        """
        N = input.shape[0]
        embedded = self.dropout(self.embedding(input))
        if self.model_type == 'RNN':
            output, hn = self.rnn(embedded)
        elif self.model_type == 'LSTM':
            output, (hn, cn) = self.rnn(embedded)
        hn_1 = self.relu(self.linear1(hn))
        hidden = torch.tanh(self.linear2(hn_1))
        if self.model_type == 'LSTM':
            hidden = (hidden, cn)

        return output, hidden
