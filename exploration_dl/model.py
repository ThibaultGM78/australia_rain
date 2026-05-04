import torch
import torch.nn as nn

class WeatherLSTM(nn.Module):
    def __init__(self, num_features, hidden_size=64, num_layers=2):
        super(WeatherLSTM, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=num_features, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
         # Synthese globale
        self.fc_common = nn.Linear(hidden_size, 64)
        
        # Sortie
        self.temp_head = nn.Linear(64, 1)
        self.rain_head = nn.Linear(64, 1)
        
    def forward(self, x):
        x = x.permute(0, 2, 1) 
        
        out, (hn, cn) = self.lstm(x)

        # utilise les deniers jour pour decider
        last_time_step = out[:, -1, :]
        
        x_common = torch.relu(self.fc_common(last_time_step))
        
        return self.temp_head(x_common), self.rain_head(x_common)

class WeatherRNN(nn.Module):
    def __init__(self, num_features, hidden_size=64, num_layers=2):
        super(WeatherRNN, self).__init__()
        
        # RNN classique (Elman RNN)
        self.rnn = nn.RNN(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            nonlinearity='relu', # 'relu' aide à éviter la disparition du gradient par rapport à 'tanh'
            dropout=0.2 if num_layers > 1 else 0
        )
        
        self.fc_common = nn.Linear(hidden_size, 64)
        self.temp_head = nn.Linear(64, 1)
        self.rain_head = nn.Linear(64, 1)
        
    def forward(self, x):
        # x shape: [batch, features, seq_len] -> [batch, seq_len, features]
        x = x.permute(0, 2, 1)
        
        # Sortie out: [batch, seq_len, hidden_size]
        out, hn = self.rnn(x)
        
        # On récupère le dernier état caché (dernier jour de la séquence)
        last_time_step = out[:, -1, :]
        
        x_common = torch.relu(self.fc_common(last_time_step))
        return self.temp_head(x_common), self.rain_head(x_common)