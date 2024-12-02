import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class ConvLSTMEmotionClassifier(nn.Module):
    def __init__(self, input_shape, num_classes):
        """
        Conv1D + LSTM 기반 모델 정의
        :param input_shape: 입력 데이터의 형태 (features, time_steps)
        :param num_classes: 출력 클래스 수
        """
        super(ConvLSTMEmotionClassifier, self).__init__()

        # Conv1D 블록 1
        self.conv1 = nn.Conv1d(input_shape[0], 1024, kernel_size=7, stride=1, padding=3)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        self.bn1 = nn.BatchNorm1d(1024)
        self.dropout1 = nn.Dropout(0.3)

        # Conv1D 블록 2
        self.conv2 = nn.Conv1d(1024, 512, kernel_size=7, stride=1, padding=3)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        self.bn2 = nn.BatchNorm1d(512)
        self.dropout2 = nn.Dropout(0.3)

        # Conv1D 블록 3
        self.conv3 = nn.Conv1d(512, 256, kernel_size=7, stride=1, padding=3)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.dropout3 = nn.Dropout(0.3)

        # LSTM 블록
        self.lstm1 = nn.LSTM(input_size=256, hidden_size=128, batch_first=True, bidirectional=False)
        self.dropout_lstm1 = nn.Dropout(0.3)
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=128, batch_first=True, bidirectional=False)
        self.dropout_lstm2 = nn.Dropout(0.3)
        self.lstm3 = nn.LSTM(input_size=128, hidden_size=128, batch_first=True, bidirectional=False)
        self.dropout_lstm3 = nn.Dropout(0.3)

        # Dense 블록
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, num_classes)

    def forward(self, x):
        # Conv1D 블록 1
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.bn1(x)
        x = self.dropout1(x)

        # Conv1D 블록 2
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.bn2(x)
        x = self.dropout2(x)

        # Conv1D 블록 3
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        x = self.bn3(x)
        x = self.dropout3(x)

        # LSTM 블록
        x = x.permute(0, 2, 1)  # Conv1D의 (batch, channels, time) -> LSTM의 (batch, time, features)
        x, _ = self.lstm1(x)
        x = self.dropout_lstm1(x)
        x, _ = self.lstm2(x)
        x = self.dropout_lstm2(x)
        x, _ = self.lstm3(x)
        x = self.dropout_lstm3(x)

        # 마지막 LSTM의 출력 중 최종 시퀀스만 사용
        x = x[:, -1, :]

        # Dense 블록
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        x = torch.relu(x)
        x = self.fc4(x)
        return x

class KConvLSTMEmotionClassifier(nn.Module):
    def __init__(self, input_shape, num_classes):
        """
        Conv1D + LSTM 기반 모델 정의
        :param input_shape: 입력 데이터의 형태 (time_steps, features)
        :param num_classes: 출력 클래스 수
        """
        super(KConvLSTMEmotionClassifier, self).__init__()

        # Conv1D 블록 1
        self.conv1 = nn.Conv1d(input_shape[0], 1024, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        self.bn1 = nn.BatchNorm1d(1024)
        self.dropout1 = nn.Dropout(0.3)

        # Conv1D 블록 2
        self.conv2 = nn.Conv1d(1024, 512, kernel_size=5, stride=1, padding=2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        self.bn2 = nn.BatchNorm1d(512)
        self.dropout2 = nn.Dropout(0.3)

        # Conv1D 블록 3
        self.conv3 = nn.Conv1d(512, 256, kernel_size=5, stride=1, padding=2)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.dropout3 = nn.Dropout(0.3)

        # LSTM 블록
        self.lstm1 = nn.LSTM(input_size=256, hidden_size=128, batch_first=True, bidirectional=False, dropout=0.3)
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=128, batch_first=True, bidirectional=False, dropout=0.3)
        self.lstm3 = nn.LSTM(input_size=128, hidden_size=128, batch_first=True, bidirectional=False, dropout=0.3)

        # Dense 블록
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, num_classes)

    def forward(self, x):
        # Conv1D 블록 1
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.bn1(x)
        x = self.dropout1(x)

        # Conv1D 블록 2
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.bn2(x)
        x = self.dropout2(x)

        # Conv1D 블록 3
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        x = self.bn3(x)
        x = self.dropout3(x)

        # LSTM 블록
        x = x.permute(0, 2, 1)  # Conv1D의 (batch, channels, time) -> LSTM의 (batch, time, features)
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x, _ = self.lstm3(x)

        # 마지막 LSTM의 출력 중 최종 시퀀스만 사용
        x = x[:, -1, :]  # LSTM의 마지막 시퀀스를 사용

        # Dense 블록
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        x = torch.relu(x)
        x = self.fc4(x)

        return x   

# CNN-only 모델
class CNNEmotionClassifier(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(CNNEmotionClassifier, self).__init__()

        # Conv1D 블록
        self.conv1 = nn.Conv1d(input_shape[0], 1024, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        self.bn1 = nn.BatchNorm1d(1024)
        self.dropout1 = nn.Dropout(0.3)

        self.conv2 = nn.Conv1d(1024, 512, kernel_size=5, stride=1, padding=2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        self.bn2 = nn.BatchNorm1d(512)
        self.dropout2 = nn.Dropout(0.3)

        self.conv3 = nn.Conv1d(512, 256, kernel_size=5, stride=1, padding=2)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.dropout3 = nn.Dropout(0.3)

        # Fully Connected Layer
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.bn1(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.bn2(x)
        x = self.dropout2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        x = self.bn3(x)
        x = self.dropout3(x)

        # Global Average Pooling
        x = torch.mean(x, dim=2)

        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        x = torch.relu(x)
        x = self.fc4(x)
        return x

class CNN_residual_EmotionClassifier(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(CNN_residual_EmotionClassifier, self).__init__()

        # Conv1D 블록
        self.conv1 = nn.Conv1d(input_shape[0], 1024, kernel_size=5, stride=1, padding=2)
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        self.bn1 = nn.BatchNorm1d(1024)
        self.dropout1 = nn.Dropout(0.3)

        self.conv2 = nn.Conv1d(1024, 512, kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        self.bn2 = nn.BatchNorm1d(512)
        self.dropout2 = nn.Dropout(0.3)

        self.conv3 = nn.Conv1d(512, 256, kernel_size=5, stride=1, padding=2)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.dropout3 = nn.Dropout(0.3)

        # Residual Connection: Adjust stride to match dimensions
        self.residual_conv = nn.Conv1d(input_shape[0], 256, kernel_size=1, stride=8, padding=0)

        # Fully Connected Layer
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, num_classes)

    def forward(self, x): #padding 사용
        x1 = self.conv1(x)
        x1 = self.relu(x1)
        x1 = self.pool1(x1)
        x1 = self.bn1(x1)
        x1 = self.dropout1(x1)

        x2 = self.conv2(x1)
        x2 = self.relu(x2)
        x2 = self.pool2(x2)
        x2 = self.bn2(x2)
        x2 = self.dropout2(x2)

        x3 = self.conv3(x2)
        x3 = self.relu(x3)
        x3 = self.pool3(x3)
        x3 = self.bn3(x3)
        x3 = self.dropout3(x3)

        # Residual Connection
        residual = self.residual_conv(x)

        # 크기 일치 (Padding)
        if x3.size(2) != residual.size(2):
            diff = abs(x3.size(2) - residual.size(2))
            if x3.size(2) > residual.size(2):  # residual에 Padding 추가
                residual = F.pad(residual, (0, diff))
            else:  # x3에 Padding 추가 (드물지만 가능)
                x3 = F.pad(x3, (0, diff))

        x3 = x3 + residual

        # Global Average Pooling
        x3 = torch.mean(x3, dim=2)

        x = self.fc1(x3)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        x = torch.relu(x)
        x = self.fc4(x)
        return x


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _ = x.size()
        y = torch.mean(x, dim=2)  # Global Average Pooling
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y)
        y = y.view(b, c, 1)
        return x * y  # Channel-wise Attention

class CNN_emotionClassifierWithSE(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(CNN_emotionClassifierWithSE, self).__init__()
        self.conv1 = nn.Conv1d(input_shape[0], 1024, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        self.bn1 = nn.BatchNorm1d(1024)
        self.se1 = SEBlock(1024)

        self.conv2 = nn.Conv1d(1024, 512, kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        self.bn2 = nn.BatchNorm1d(512)
        self.se2 = SEBlock(512)

        self.conv3 = nn.Conv1d(512, 256, kernel_size=5, stride=1, padding=2)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.se3 = SEBlock(256)

        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.bn1(x)
        x = self.se1(x)

        x = self.conv2(x)
        x = self.pool2(x)
        x = self.bn2(x)
        x = self.se2(x)

        x = self.conv3(x)
        x = self.pool3(x)
        x = self.bn3(x)
        x = self.se3(x)

        # Global Average Pooling
        x = torch.mean(x, dim=2)

        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        return x


class LSTMEmotionClassifier(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(LSTMEmotionClassifier, self).__init__()

        self.lstm1 = nn.LSTM(input_size=input_shape[0], hidden_size=128, batch_first=True, bidirectional=False)
        self.dropout_lstm1 = nn.Dropout(0.3)
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=128, batch_first=True, bidirectional=False)
        self.dropout_lstm2 = nn.Dropout(0.3)

        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (batch, features, time) -> (batch, time, features)
        x, _ = self.lstm1(x)
        x = self.dropout_lstm1(x)
        x, _ = self.lstm2(x)
        x = self.dropout_lstm2(x)

        # 마지막 시퀀스 출력만 사용
        x = x[:, -1, :]

        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        x = torch.relu(x)
        x = self.fc4(x)
        return x

class CNNGRUEmotionClassifier(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(CNNGRUEmotionClassifier, self).__init__()

        # Conv1D 블록
        self.conv1 = nn.Conv1d(input_shape[0], 1024, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        self.bn1 = nn.BatchNorm1d(1024)
        self.dropout1 = nn.Dropout(0.3)

        self.conv2 = nn.Conv1d(1024, 512, kernel_size=5, stride=1, padding=2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        self.bn2 = nn.BatchNorm1d(512)
        self.dropout2 = nn.Dropout(0.3)

        self.conv3 = nn.Conv1d(512, 256, kernel_size=5, stride=1, padding=2)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.dropout3 = nn.Dropout(0.3)

        # GRU 블록
        self.gru = nn.GRU(input_size=256, hidden_size=128, batch_first=True, bidirectional=False)
        self.dropout_gru = nn.Dropout(0.3)

        # Fully Connected Layer
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, num_classes)

    def forward(self, x):
        # Conv1D 처리
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.bn1(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.bn2(x)
        x = self.dropout2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        x = self.bn3(x)
        x = self.dropout3(x)

        # GRU 처리
        x = x.permute(0, 2, 1)  # (batch, channels, time) -> (batch, time, features)
        x, _ = self.gru(x)
        x = self.dropout_gru(x)

        # 마지막 시퀀스 출력만 사용
        x = x[:, -1, :]

        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        x = torch.relu(x)
        x = self.fc4(x)
        return x

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
            nn.Conv1d(features, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(0.3)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
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

class GRUEmotionClassifier(nn.Module):
    def __init__(self, input_shape, num_classes, hidden_size=128, num_layers=3, dropout=0.3):
        """
        GRU 기반 감정 분류 모델
        :param input_shape: (features, time_steps) 입력 데이터 차원
        :param num_classes: 분류할 클래스 수
        :param hidden_size: GRU의 히든 레이어 크기
        :param num_layers: GRU 레이어 수
        :param dropout: 드롭아웃 비율
        """
        super(GRUEmotionClassifier, self).__init__()
        features, time_steps = input_shape
        
        # GRU 블록
        self.gru = nn.GRU(features, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        
        # Fully Connected 블록
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        # GRU는 (batch, time_steps, features) 형태의 입력을 기대
        x = x.permute(0, 2, 1)  # (batch, features, time_steps) -> (batch, time_steps, features)
        _, h_n = self.gru(x)  # GRU의 마지막 히든 스테이트 h_n을 사용
        x = h_n[-1]  # 마지막 레이어의 히든 상태
        x = self.fc(x)  # Fully Connected 레이어로 분류
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

class TransformerEmotionClassifier(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(TransformerEmotionClassifier, self).__init__()
        self.embedding = nn.Linear(input_shape[0], 256)  # Embedding layer
        self.positional_encoding = PositionalEncoding(d_model=256)
        encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8, dim_feedforward=512)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (batch, features, time) -> (batch, time, features)
        x = self.embedding(x)
        x = self.positional_encoding(x)  # Add positional encoding
        x = self.transformer_encoder(x)
        x = x[:, -1, :]  # Use the last time step
        x = self.fc(x)
        return x