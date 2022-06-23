import numpy as np
import torch
from torch import nn
import random

class TransformerTranslator(nn.Module):
    """
    A Transformer which encodes a sequence of text and performs binary classification.
    """
    def __init__(self, input_size, output_size, device, hidden_dim=128, num_heads=2, dim_feedforward=2048, dim_k=96, dim_v=96, dim_q=96, max_length=43):
        """
        :param input_size: the size of the input, which equals to the number of words in source language vocabulary
        :param output_size: the size of the output, which equals to the number of words in target language vocabulary
        :param hidden_dim: the dimensionality of the output embeddings that go into the final layer
        :param num_heads: the number of Transformer heads to use
        :param dim_feedforward: the dimension of the feedforward network model
        :param dim_k: the dimensionality of the key vectors
        :param dim_q: the dimensionality of the query vectors
        :param dim_v: the dimensionality of the value vectors
        """
        super(TransformerTranslator, self).__init__()
        assert hidden_dim % num_heads == 0
        
        self.num_heads = num_heads
        self.word_embedding_dim = hidden_dim
        self.hidden_dim = hidden_dim
        self.dim_feedforward = dim_feedforward
        self.max_length = max_length
        self.input_size = input_size
        self.output_size = output_size
        self.device = device
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.dim_q = dim_q
        
        seed_torch(0)
        

        # Initialize the embedding lookup

        self.tok = nn.Embedding(self.input_size, self.hidden_dim)
        self.pos = nn.Embedding(self.max_length, self.hidden_dim)
        if torch.cuda.is_available():
            self.tok = self.tok.to(device)
            self.pos = self.pos.to(device)

        # Initializations for multi-head self-attention.
        
        # Head #1
        self.k1 = nn.Linear(self.hidden_dim, self.dim_k)
        self.v1 = nn.Linear(self.hidden_dim, self.dim_v)
        self.q1 = nn.Linear(self.hidden_dim, self.dim_q)
        
        # Head #2
        self.k2 = nn.Linear(self.hidden_dim, self.dim_k)
        self.v2 = nn.Linear(self.hidden_dim, self.dim_v)
        self.q2 = nn.Linear(self.hidden_dim, self.dim_q)
        
        self.softmax = nn.Softmax(dim=2)
        self.attention_head_projection = nn.Linear(self.dim_v * self.num_heads, self.hidden_dim)
        self.norm_mh = nn.LayerNorm(self.hidden_dim)
        
        if torch.cuda.is_available():
            self.k1 = self.k1.to(device)
            self.v1 = self.v1.to(device)
            self.q1 = self.q1.to(device)
            self.k2 = self.k2.to(device)
            self.v2 = self.v2.to(device)
            self.q2 = self.q2.to(device)
            self.softmax = self.softmax.to(device)
            self.attention_head_projection = self.attention_head_projection.to(device)
            self.norm_mh = self.norm_mh.to(device)

        # Initialize what you need for the feed-forward layer.
        self.ff_1 = nn.Linear(self.hidden_dim, self.dim_feedforward)
        self.relu = nn.ReLU()
        self.ff_2 = nn.Linear(self.dim_feedforward, self.hidden_dim)
        self.ff_norm = nn.LayerNorm(self.hidden_dim)
        
        if torch.cuda.is_available():
            self.ff_1 = self.ff_1.to(device)
            self.ff_2 = self.ff_2.to(device)
            self.relu = self.relu.to(device)
            self.ff_norm = self.ff_norm.to(device)

        # Initialize what you need for the final layer (1-2 lines).
        self.finalLayer = nn.Linear(self.hidden_dim, self.output_size)
        
        if torch.cuda.is_available():
            self.finalLayer = self.finalLayer.to(device)

    def forward(self, inputs):
        """
        This function computes the full Transformer forward pass.
        """

        # Implement the full Transformer stack for the forward pass.
        inputs = inputs.to(self.device)
        embedded = self.embed(inputs)
        hidden_states = self.multi_head_attention(embedded)
        FFoutputs = self.feedforward_layer(hidden_states)
        outputs = self.final_layer(FFoutputs)
        
        if torch.cuda.is_available():
            outputs = outputs.to(self.device)
        return outputs
    
    
    def embed(self, inputs):
        """
        :param inputs: intTensor of shape (N,T)
        :returns embeddings: floatTensor of shape (N,T,H)
        """
        # Implement the embedding lookup.
        N, T = inputs.shape
        tok = self.tok(inputs)
        pos = self.pos(torch.arange(0,T))
        embeddings = tok+pos
        
        if torch.cuda.is_available():
            embeddings = embeddings.to(self.device)
        return embeddings
        
    def multi_head_attention(self, inputs):
        """
        :param inputs: float32 Tensor of shape (N,T,H)
        :returns outputs: float32 Tensor of shape (N,T,H)
        """
        # Implement multi-head self-attention followed by add + norm.
        Q1 = self.q1(inputs)
        K1 = self.k1(inputs)
        V1 = self.v1(inputs)
        
        att1 = torch.matmul(self.softmax(torch.matmul(Q1, K1.transpose(-2, -1)) / np.sqrt(self.dim_k)), V1)

        Q2 = self.q2(inputs)
        K2 = self.k2(inputs)
        V2 = self.v2(inputs)
        att2 = torch.matmul(self.softmax(torch.matmul(Q2, K2.transpose(-2, -1)) / np.sqrt(self.dim_k)), V2)

        head = torch.cat((att1, att2), dim=2)
        outputs = self.attention_head_projection(head)
        outputs = self.norm_mh(torch.add(outputs, inputs))
        
        if torch.cuda.is_available():
            outputs = outputs.to(self.device)
        return outputs
    
    def feedforward_layer(self, inputs):
        """
        :param inputs: float32 Tensor of shape (N,T,H)
        :returns outputs: float32 Tensor of shape (N,T,H)
        """
        # Implement the feedforward layer followed by add + norm.
        ff1_out = self.ff_1(inputs)
        relu_out = self.relu(ff1_out)
        ff2_out = self.ff_2(relu_out)
        outputs = self.ff_norm(ff2_out + inputs)
        
        if torch.cuda.is_available():
            outputs = outputs.to(self.device)
        return outputs
        
    
    def final_layer(self, inputs):
        """
        :param inputs: float32 Tensor of shape (N,T,H)
        :returns outputs: float32 Tensor of shape (N,T,V)
        """
        # the final layer for the Transformer Translator.
        outputs = self.finalLayer(inputs)
        
        if torch.cuda.is_available():
            outputs = outputs.to(self.device)
        return outputs
        

def seed_torch(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True