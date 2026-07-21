import torch
import torch.nn as nn
from config import n_embd

class LSTMModel(nn.Module):

    def __init__(self,vocab_size,dropout):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size,n_embd)
        self.lstm = nn.LSTM(input_size = n_embd,
                            hidden_size = 512,
                            num_layers = 2,
                            dropout = dropout,
                            batch_first = True
        )

        self.dropout = nn.Dropout(dropout)
        self.fully_c = nn.Linear(512,vocab_size)

    def forward(self,x):
        x = self.embedding(x)
        x = self.dropout(x)

        intermediate ,(final_h,final_c) = self.lstm(x)
        x = self.dropout(x)
        logits = self.fully_c(final_h[-1]) # (batch,seq_len,vocab_size)
        return logits
