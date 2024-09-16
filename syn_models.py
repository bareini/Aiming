import torch
import torch.nn as nn

class SIMLSTM(nn.Module):
    def __init__(self, input_dim=9, hidden_dim=10, encoding=3, drop_prob=0.05, device='cpu'):
        super(SIMLSTM, self).__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_size=encoding, hidden_size=hidden_dim, num_layers=3)
        self.dropout = nn.Dropout(drop_prob)
        self.relu = nn.ReLU()
        self.encoding = nn.Linear(input_dim, encoding)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, input):
        out = self.encoding(input.float())
        lstm_out, _ = self.lstm(out)
        out = self.fc(lstm_out)
        return out.squeeze()

class LR(nn.Module):
    def __init__(self, input_dim=9, final_out=1, device='cpu'):
        super(LR, self).__init__()
        self.device = device
        self.fc = nn.Linear(input_dim, final_out)

    def forward(self, input):
        out = self.fc(input.float())
        return out.squeeze()
