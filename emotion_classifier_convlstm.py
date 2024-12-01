import torch
import torch.nn as nn

class ConvLSTMEmotionClassifier(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(ConvLSTMEmotionClassifier, self).__init__()
        self.conv1 = nn.Conv1d(input_shape[0], 1024, kernel_size=7, stride=1, padding=3)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        self.bn1 = nn.BatchNorm1d(1024)
        self.dropout1 = nn.Dropout(0.3)

        self.conv2 = nn.Conv1d(1024, 512, kernel_size=7, stride=1, padding=3)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        self.bn2 = nn.BatchNorm1d(512)
        self.dropout2 = nn.Dropout(0.3)

        self.conv3 = nn.Conv1d(512, 256, kernel_size=7, stride=1, padding=3)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.dropout3 = nn.Dropout(0.3)

        self.lstm1 = nn.LSTM(input_size=256, hidden_size=128, batch_first=True, bidirectional=False)
        self.dropout_lstm1 = nn.Dropout(0.3)

        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

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

        x = x.permute(0, 2, 1)
        x, _ = self.lstm1(x)
        x = self.dropout_lstm1(x)

        x = x[:, -1, :]
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        return x
    