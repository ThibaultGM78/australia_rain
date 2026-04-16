import torch
import torch.nn as nn

class WeatherCNN(nn.Module):
    def __init__(self, num_features):
        super(WeatherCNN, self).__init__()
    
        self.conv_block = nn.Sequential(
            # Conv1d (128) : Detecte motifs meteo locaux
            nn.Conv1d(num_features, 128, kernel_size=3, padding=1),
            # ReLU : Garde signaux positifs importants
            nn.ReLU(),
            # BatchNorm : Stabilise l'apprentissage du modele
            nn.BatchNorm1d(128),
            # MaxPool : Reduit donnees, garde l'essentiel
            nn.MaxPool1d(2),
            # Conv1d (64) : Affine les motifs complexes
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            # ReLU : Garde signaux positifs importants 
            nn.ReLU(),
            # Flatten : Aplatit pour couches finales.
            nn.Flatten()
        )
        
        # Synthese globale
        self.fc_common = nn.Linear(64 * 3, 128)
        
        # Sortie
        self.temp_head = nn.Linear(128, 1)
        self.rain_head = nn.Linear(128, 1)
        
    def forward(self, x):
        x = self.conv_block(x)
        x = torch.relu(self.fc_common(x))
        return self.temp_head(x), self.rain_head(x)

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