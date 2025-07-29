import os
import numpy as np
import torch
from torch import nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib
from data_preprocessing import load_and_preprocess_data

SEQ_LENGTH = 30
EPOCHS = 100
BATCH_SIZE = 32

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

def create_sequences(data, seq_length):
    sequences = []
    labels = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i+seq_length])
        labels.append(data[i+seq_length])
    return np.array(sequences), np.array(labels)

def main():
    data_path = os.path.join(os.path.dirname(__file__), '../data/btcusd_1-min_data.csv')
    df = load_and_preprocess_data(data_path)
    prices = df[['Close']].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    prices_scaled = scaler.fit_transform(prices)
    X, y = create_sequences(prices_scaled, SEQ_LENGTH)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    # Add feature dimension
    X_train = X_train.unsqueeze(-1)
    y_train = y_train.unsqueeze(-1)
    model = TransformerPricePredictor(input_dim=1, d_model=32, nhead=2, num_layers=1)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item():.6f}')
    # Save model and scaler
    os.makedirs('../models', exist_ok=True)
    torch.save(model.state_dict(), '../models/transformer_model.pth')
    joblib.dump(scaler, '../models/transformer_scaler.save')
    print('Transformer model and scaler saved.')

if __name__ == "__main__":
    main() 