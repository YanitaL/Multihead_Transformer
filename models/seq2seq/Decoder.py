import random
import torch
import torch.nn as nn
import torch.optim as optim

class Decoder(nn.Module):
    """ The Decoder module of the Seq2Seq model 
        You will need to complete the init function and the forward function.
    """

    def __init__(self, emb_size, encoder_hidden_size, decoder_hidden_size, output_size, dropout=0.2, model_type="RNN"):
        super(Decoder, self).__init__()

        self.emb_size = emb_size
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.output_size = output_size
        self.model_type = model_type
        self.embedding = nn.Embedding(self.output_size,self.emb_size)
        
        if self.model_type == 'RNN':
            self.rnn = nn.RNN(self.emb_size, self.decoder_hidden_size, batch_first=True, dropout=dropout)
        elif self.model_type == 'LSTM':
            self.rnn = nn.LSTM(self.emb_size, self.decoder_hidden_size, batch_first=True, dropout=dropout)
        
        self.linear = nn.Linear(self.decoder_hidden_size, self.output_size)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden):
        """ The forward pass of the decoder
            Args:
                input (tensor): the encoded sequences of shape (batch_size, 1)
                hidden (tensor): the hidden weights of the previous time step from the decoder
            Returns:
                output (tensor): the output of the decoder
                hidden (tensor): the weights coming out of the hidden unit
        """
        embedded = self.dropout(self.embedding(input))

        if self.model_type == 'RNN':
            outputs, hn = self.rnn(embedded,hidden)
        elif self.model_type == 'LSTM':
            outputs, (hn, cn) = self.rnn(embedded,hidden)
            
        outputs = outputs[:, 0, :]
        output = self.logsoftmax(self.linear(outputs))
        
        if self.model_type == 'RNN':
            hidden = hn
        elif self.model_type == 'LSTM':
            hidden = (hn, cn)

        return output, hidden
