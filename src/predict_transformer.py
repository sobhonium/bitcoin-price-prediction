import os
import numpy as np
import torch
from torch import nn
import joblib
from data_preprocessing import load_and_preprocess_data

SEQ_LENGTH = 30

class TransformerPricePredictor(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dropout=0.1):
        super(TransformerPricePredictor, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.positional_encoding = nn.Parameter(torch.zeros(1, SEQ_LENGTH, d_model))
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead, num_encoder_layers=num_layers, num_decoder_layers=num_layers, dropout=dropout
        )
        self.fc_out = nn.Linear(d_model, 1)

    def forward(self, src):
        src = self.embedding(src) + self.positional_encoding
        src = src.permute(1, 0, 2)
        output = self.transformer(src, src)
        output = output.permute(1, 0, 2)
        return self.fc_out(output[:, -1, :])

def main():
    data_path = os.path.join(os.path.dirname(__file__), '../data/btcusd_1-min_data.csv')
    df = load_and_preprocess_data(data_path)
    prices = df[['Close']].values
    scaler = joblib.load('../models/transformer_scaler.save')
    prices_scaled = scaler.transform(prices)
    X_test_future = prices_scaled[-SEQ_LENGTH:]
    X_test_future = torch.tensor(X_test_future, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
    model = TransformerPricePredictor(input_dim=1, d_model=32, nhead=2, num_layers=1)
    model.load_state_dict(torch.load('../models/transformer_model.pth'))
    model.eval()
    y_pred = model(X_test_future).detach().numpy()
    future_price = scaler.inverse_transform(y_pred.reshape(-1, 1))
    print(f"Predicted Bitcoin Price for Tomorrow (Transformer): {future_price[0][0]}")

if __name__ == "__main__":
    main() 