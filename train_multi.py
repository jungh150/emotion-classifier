import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from utils import load_data, preprocess_data
import numpy as np
import os

from emotion_classifier_convlstm import ConvLSTMEmotionClassifier
from emotion_classifier_cnn import CNNEmotionClassifier
from emotion_classifier_lstm import LSTMEmotionClassifier
from emotion_classifier_transformer import TransformerEmotionClassifier
from emotion_classifier_gru import GRUEmotionClassifier
from emotion_classifier_rnn import RNNEmotionClassifier
from emotion_classifier_crnn import CRNNEmotionClassifier

def plot_multiple_models_curves(models_results, num_epochs):
    """여러 모델의 Accuracy와 Loss를 각각 별도의 그래프에 그리기"""
    epochs = range(1, num_epochs + 1)

    # Accuracy Plot
    plt.figure(figsize=(14, 6))
    for model_name, results in models_results.items():
        train_accuracies, val_accuracies, _, _ = results
        plt.plot(epochs, train_accuracies, label=f'{model_name} Train Accuracy', linestyle='-', marker='x')
        plt.plot(epochs, val_accuracies, label=f'{model_name} Validation Accuracy', linestyle='-', marker='o')
    plt.title('Accuracy Curves for Multiple Models')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='best')
    plt.grid()
    plt.show()

    # Loss Plot
    plt.figure(figsize=(14, 6))
    for model_name, results in models_results.items():
        _, _, train_losses, val_losses = results
        plt.plot(epochs, train_losses, label=f'{model_name} Train Loss', linestyle='-', marker='x')
        plt.plot(epochs, val_losses, label=f'{model_name} Validation Loss', linestyle='-', marker='o')
    plt.title('Loss Curves for Multiple Models')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='best')
    plt.grid()
    plt.show()

def train_multiple_models(models, X_train, y_train, X_val, y_val, config):
    """여러 모델을 학습시키고 결과를 저장"""
    num_epochs = config['num_epochs']
    batch_size = config['batch_size']
    learning_rate = config['learning_rate']
    device = config['device']

    models_results = {}

    for model_name, model_class in models.items():
        print(f"Training {model_name}...")
        model = model_class((X_train.size(1), X_train.size(2)), len(np.unique(y_train))).to(device)

        # 손실 함수와 옵티마이저 정의
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        train_accuracies = []
        val_accuracies = []
        train_losses = []
        val_losses = []

        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            total_correct = 0
            total_samples = 0

            # Train loop
            for i in range(0, X_train.size(0), batch_size):
                X_batch = X_train[i:i + batch_size].to(device)
                y_batch = y_train[i:i + batch_size].to(device)

                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                total_correct += (preds == y_batch).sum().item()
                total_samples += y_batch.size(0)

            train_accuracy = total_correct / total_samples
            train_loss = running_loss / (X_train.size(0) // batch_size)
            train_accuracies.append(train_accuracy)
            train_losses.append(train_loss)

            # Validation loop
            model.eval()
            val_loss = 0.0
            total_correct = 0
            total_samples = 0
            with torch.no_grad():
                for i in range(0, X_val.size(0), batch_size):
                    X_batch = X_val[i:i + batch_size].to(device)
                    y_batch = y_val[i:i + batch_size].to(device)

                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    val_loss += loss.item()

                    _, preds = torch.max(outputs, 1)
                    total_correct += (preds == y_batch).sum().item()
                    total_samples += y_batch.size(0)

            val_accuracy = total_correct / total_samples
            val_loss /= (X_val.size(0) // batch_size)
            val_accuracies.append(val_accuracy)
            val_losses.append(val_loss)

            print(f"[{model_name}] Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

        # Store results for the model
        models_results[model_name] = (train_accuracies, val_accuracies, train_losses, val_losses)

    return models_results

def main():
    # 데이터 로드 및 전처리
    csv_path = 'datasets/emotion_melpath_dataset.csv'
    file_paths, labels = load_data(csv_path)
    X, y = preprocess_data(file_paths, labels)

    # 학습 및 검증 데이터 분리
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # 데이터를 Tensor 형태로 변환
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).permute(0, 2, 1)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).permute(0, 2, 1)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)

    # 모델 선택 (여러 모델 비교 가능)
    models = {
        "convlstm": ConvLSTMEmotionClassifier,
        "cnn": CNNEmotionClassifier,
        "lstm": LSTMEmotionClassifier,
        "gru": GRUEmotionClassifier,
        "rnn": RNNEmotionClassifier,
        "crnn": CRNNEmotionClassifier
    }

    # 학습 설정
    config = {
        'num_epochs': 120,
        'batch_size': 32,
        'learning_rate': 0.001,
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    }

    # 여러 모델 학습
    models_results = train_multiple_models(models, X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, config)

    # 그래프 그리기
    plot_multiple_models_curves(models_results, config['num_epochs'])

if __name__ == "__main__":
    main()
