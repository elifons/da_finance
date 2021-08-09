import torch
import torch.nn as nn

class LSTMNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, device, drop_prob=0.1, seed=4):
        super(LSTMNet, self).__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.seed = torch.manual_seed(seed)
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True,  dropout=drop_prob)
        self.drop2 = nn.Dropout(drop_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        h_ = self.init_hidden(x)
        out, h = self.lstm(x, h_)
        out = self.drop2(out)
        out = self.fc(out[:, -1])  
        return out
    
    def init_hidden(self, x):
        h0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim).to(self.device)
        c0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim).to(self.device)
        return (h0,c0)
        # weight = next(self.parameters()).data
        # hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device),
        #           weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device))
        # return hidden