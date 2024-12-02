import torch
import torch.nn as nn

class RNNEmotionClassifier(nn.Module):
    def __init__(self, input_shape, num_classes, hidden_size=128, num_layers=3, dropout=0.3):
        """
        RNN 기반 감정 분류 모델
        :param input_shape: (features, time_steps) 입력 데이터 차원
        :param num_classes: 분류할 클래스 수
        :param hidden_size: RNN의 히든 레이어 크기
        :param num_layers: RNN 레이어 수
        :param dropout: 드롭아웃 비율
        """
        super(RNNEmotionClassifier, self).__init__()
        features, time_steps = input_shape

        # RNN 블록
        self.rnn = nn.RNN(features, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)

        # Fully Connected 블록
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        # RNN은 (batch, time_steps, features) 형태의 입력을 기대
        x = x.permute(0, 2, 1)  # (batch, features, time_steps) -> (batch, time_steps, features)
        _, h_n = self.rnn(x)  # 마지막 히든 스테이트 h_n을 사용
        x = h_n[-1]  # 마지막 레이어의 히든 상태
        x = self.fc(x)  # Fully Connected 레이어로 분류
        return x
