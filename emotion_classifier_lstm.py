import torch
import torch.nn as nn

class LSTMEmotionClassifier(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(LSTMEmotionClassifier, self).__init__()
        features, time_steps = input_shape
        
        # LSTM 블록
        self.lstm = nn.LSTM(features, 128, num_layers=3, batch_first=True, dropout=0.3)
        
        # Fully Connected 블록
        self.fc = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        # LSTM은 (batch, time_steps, features) 형태의 입력을 기대
        x = x.permute(0, 2, 1)  # (batch, features, time_steps) -> (batch, time_steps, features)
        x, _ = self.lstm(x)  # LSTM의 출력
        x = x[:, -1, :]  # 마지막 time step의 출력 사용
        x = self.fc(x)  # Fully Connected 레이어로 분류
        return x
