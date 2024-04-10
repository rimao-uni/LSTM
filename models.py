import torch
import torch.nn as nn

class CustomLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(CustomLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Forget Gate
        self.Wf = nn.Parameter(torch.Tensor(input_size + hidden_size, hidden_size))
        self.bf = nn.Parameter(torch.Tensor(hidden_size))
        
        # Input Gate
        self.Wi = nn.Parameter(torch.Tensor(input_size + hidden_size, hidden_size))
        self.bi = nn.Parameter(torch.Tensor(hidden_size))
        
        # Candidate Layer
        self.Wc = nn.Parameter(torch.Tensor(input_size + hidden_size, hidden_size))
        self.bc = nn.Parameter(torch.Tensor(hidden_size))
        
        # Output Gate
        self.Wo = nn.Parameter(torch.Tensor(input_size + hidden_size, hidden_size))
        self.bo = nn.Parameter(torch.Tensor(hidden_size))
        
        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)

    def forward(self, x, hidden_state):
        h_prev, c_prev = hidden_state
        
        # Concatenate input and previous hidden state
        combined = torch.cat((x, h_prev), dim=1)
        f = torch.sigmoid(torch.matmul(combined, self.Wf) + self.bf)
        i = torch.sigmoid(torch.matmul(combined, self.Wi) + self.bi)
        c_tilde = torch.tanh(torch.matmul(combined, self.Wc) + self.bc)
        c = f * c_prev + i * c_tilde
        
        # Output Gate
        o = torch.sigmoid(torch.matmul(combined, self.Wo) + self.bo)
        
        # Update hidden state
        h = o * torch.tanh(c)
        
        return h, c

class LSTM(nn.Module):
    def __init__(self,config):
        super(LSTM, self).__init__()
        self.config_model = config['model']
        self.input_size  = self.config_model['input_size']
        self.hidden_size = self.config_model['hidden_size']
        self.output_size = self.config_model['output_size']

        self.lstm_cell = CustomLSTMCell(self.input_size, self.hidden_size)
        self.fc = nn.Linear(self.hidden_size, self.output_size)
        self.act = nn.LeakyReLU()
        self.dropout = nn.Dropout(self.config_model['dropout_rate'])  # Dropout rate

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        
        # Initialize hidden state and cell state
        h = torch.zeros(batch_size, self.hidden_size).to(x.device)
        c = torch.zeros(batch_size, self.hidden_size).to(x.device)

        # Iterate over sequence
        for t in range(seq_len):
            h, c = self.lstm_cell(x[:, t, :], (h, c))
        
        # Apply dropout
        h = self.dropout(h)
        
        # Fully connected layer
        out = self.fc(h)
        
        # Activation function
        out = self.act(out)

        return out
