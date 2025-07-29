import os
import numpy as np
import torch
from torch import nn
import joblib
from data_preprocessing import load_and_preprocess_data

SEQ_LENGTH = 30

class BitcoinLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2):
        super(BitcoinLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

def main():
    data_path = os.path.join(os.path.dirname(__file__), '../data/btcusd_1-min_data.csv')
    df = load_and_preprocess_data(data_path)
    prices = df[['Close']].values
    scaler = joblib.load('../models/lstm_scaler.save')
    prices_scaled = scaler.transform(prices)
    X_test_future = prices_scaled[-SEQ_LENGTH:]
    X_test_future = torch.tensor(X_test_future, dtype=torch.float32).unsqueeze(0)
    model = BitcoinLSTM()
    model.load_state_dict(torch.load('../models/lstm_model.pth'))
    model.eval()
    y_pred = model(X_test_future).detach().numpy()
    future_price = scaler.inverse_transform(y_pred.reshape(-1, 1))
    print(f"Predicted Bitcoin Price for Tomorrow: {future_price[0][0]}")

if __name__ == "__main__":
    main() 