import torch
import torch.nn as nn

class ConvLSTMEmotionClassifier(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(ConvLSTMEmotionClassifier, self).__init__()
        features, time_steps = input_shape
        
        # Conv1D 블록 1
        self.conv1 = nn.Sequential(
            nn.Conv1d(features, 1024, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.3)
        )
        # Conv1D 블록 2
        self.conv2 = nn.Sequential(
            nn.Conv1d(1024, 512, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3)
        )
        # Conv1D 블록 3
        self.conv3 = nn.Sequential(
            nn.Conv1d(512, 256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3)
        )
        
        # LSTM 블록
        self.lstm = nn.LSTM(256, 128, num_layers=3, batch_first=True, dropout=0.3)
        
        # Attention Mechanism 추가
        self.attention = nn.Sequential(
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
            nn.Softmax(dim=1)
        )
        
        # Fully Connected 블록
        self.fc = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        # LSTM에 입력하기 위해 (batch, time_steps, features) 형태로 변환
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        
        # Attention Mechanism 적용
        attn_weights = self.attention(x)  # (batch, time_steps, 1)
        x = torch.sum(attn_weights * x, dim=1)  # 가중치 적용 후 time_steps 축소
        
        x = self.fc(x)
        return x
