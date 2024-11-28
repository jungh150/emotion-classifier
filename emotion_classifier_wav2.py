import torchaudio.models
import torch.nn as nn

class Wav2Vec2EmotionClassifier(nn.Module):
    def __init__(self, input_shape=None, num_classes=10):
        super(Wav2Vec2EmotionClassifier, self).__init__()
        # Pretrained Wav2Vec2.0 모델 로드
        self.wav2vec2 = torchaudio.models.wav2vec2_base()
        
        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),  # Wav2Vec2.0 출력 크기 512
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        # Wav2Vec2.0로 특징 추출
        x = self.wav2vec2(x)
        x = x.mean(dim=1)  # Time dimension을 평균 처리
        x = self.classifier(x)
        return x
