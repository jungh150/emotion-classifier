import torch
import torch.nn as nn

class TransformerEmotionClassifier(nn.Module):
    def __init__(self, input_shape, num_classes, d_model=128, nhead=4, num_layers=3, dim_feedforward=512, dropout=0.3):
        """
        Transformer 기반 감정 분류 모델
        :param input_shape: (features, time_steps) 입력 데이터 차원
        :param num_classes: 분류할 클래스 수
        :param d_model: Transformer의 임베딩 차원
        :param nhead: 멀티 헤드 어텐션의 헤드 수
        :param num_layers: Transformer 인코더 레이어 수
        :param dim_feedforward: 피드포워드 네트워크 차원
        :param dropout: 드롭아웃 비율
        """
        super(TransformerEmotionClassifier, self).__init__()
        features, time_steps = input_shape

        # Linear Embedding: Conv1D처럼 feature를 d_model 크기로 확장
        self.embedding = nn.Linear(features, d_model)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # (batch, time_steps, features)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Fully Connected 블록
        self.fc = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        # (batch, features, time_steps) -> (batch, time_steps, features)
        x = x.permute(0, 2, 1)
        
        # Embedding을 통해 d_model 차원으로 변환
        x = self.embedding(x)
        
        # Transformer Encoder
        x = self.transformer(x)

        # 마지막 time step의 출력 사용
        x = x[:, -1, :]
        
        # Fully Connected 레이어로 분류
        x = self.fc(x)
        return x
