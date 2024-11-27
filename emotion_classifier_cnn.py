import torch.nn as nn

class CNNEmotionClassifier(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(CNNEmotionClassifier, self).__init__()
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
        
        # Fully Connected 블록
        self.fc = nn.Sequential(
            nn.Flatten(),  # Conv1D 출력 (batch, features, time_steps)을 펼치기
            nn.Linear(256 * (time_steps // 8), 128),  # Time steps 크기는 MaxPooling의 영향으로 축소
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        # Conv1D 블록을 통과
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        # Fully Connected 레이어로 입력
        x = self.fc(x)
        return x
