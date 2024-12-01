import torch
import torch.nn as nn

class CRNNEmotionClassifier(nn.Module):
    def __init__(self, input_shape, num_classes, hidden_size=128, num_layers=3, dropout=0.3):
        """
        CRNN 기반 감정 분류 모델
        :param input_shape: (features, time_steps) 입력 데이터 차원
        :param num_classes: 분류할 클래스 수
        :param hidden_size: RNN의 히든 레이어 크기
        :param num_layers: RNN 레이어 수
        :param dropout: 드롭아웃 비율
        """
        super(CRNNEmotionClassifier, self).__init__()
        features, time_steps = input_shape

        # CNN 블록
        self.conv1 = nn.Sequential(
            nn.Conv1d(features, 128, kernel_size=7, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(0.3)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=7, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(0.3)
        )

        # RNN 블록
        self.rnn = nn.GRU(256, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)

        # Fully Connected 블록
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        # CNN은 (batch, features, time_steps) 형태의 입력을 기대
        x = self.conv1(x)
        x = self.conv2(x)

        # CNN 출력 (batch, channels, time_steps)을 RNN 입력 (batch, time_steps, features)로 변환
        x = x.permute(0, 2, 1)
        _, h_n = self.rnn(x)  # GRU의 마지막 히든 스테이트
        x = h_n[-1]  # 마지막 레이어의 히든 상태
        x = self.fc(x)  # Fully Connected 레이어로 분류
        return x
