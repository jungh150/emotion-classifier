import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from src.emotion_classifier.model.emotion_classifier_convlstm import ConvLSTMEmotionClassifier  # Assuming the model is defined here
from utils import load_data, preprocess_data
import numpy as np

# Add Gaussian Noise as augmentation
def add_noise(X, noise_factor=0.19):
    """Add Gaussian noise to the input data"""
    noise = np.random.normal(loc=0.0, scale=noise_factor, size=X.shape)
    return X + noise

# Shift data (time-shift augmentation)
def shift_data(X, shift_range=8):
    """Shift the data by a random number of steps"""
    shift = np.random.randint(-shift_range, shift_range)
    return np.roll(X, shift, axis=1)

def calculate_accuracy(y_true, y_pred):
    """Accuracy 계산 함수"""
    correct = (y_true == y_pred).sum().item()
    total = len(y_true)
    return correct / total

def plot_confusion_matrix(labels, preds, num_classes, save_path="confusion_matrix_best_no4.png"):
    """혼동 행렬 시각화 함수 (PNG로 저장)"""
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(num_classes), yticklabels=range(num_classes))
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')  # 혼동 행렬을 PNG로 저장
    plt.close()  # 그래프를 닫아 메모리 해제
    print(f"Confusion matrix saved as {save_path}")

# 메인 함수에서 피치 이동 증강을 추가
def main():
    # config 설정
    config = {
        'num_epochs': 150,
        'batch_size': 64,
        'learning_rate': 0.0001,
        'device': device
    }

    # 데이터 로드 및 전처리
    csv_path = '/home/smk11602/TTS/EmotionModel/accurtest/emotion3_dataset_no4.csv'
    file_paths, labels = load_data(csv_path)
    X, y = preprocess_data(file_paths, labels)

    # Hyperparameters for augmentations
    noise_factors = [0.10, 0.13, 0.16, 0.1]  # Different levels of noise
    shift_ranges = [3, 5, 7,10]  # Different shift ranges

    best_overall_accuracy = 0.0
    best_combination = None
    best_epoch = 0
    best_preds = []
    best_labels = []

    # Loop over different augmentations
    for noise_factor in noise_factors:
        for shift_range in shift_ranges:
            print(f"Training with noise_factor={noise_factor}, shift_range={shift_range}")

            # Apply augmentations
            X_augmented = add_noise(X, noise_factor=noise_factor)
            X_augmented = shift_data(X_augmented, shift_range=shift_range)

            # 레이블 값이 새로 줄어든 클래스에 맞게 재조정
            unique_labels = np.unique(y)
            label_map = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
            y_augmented = np.array([label_map[label] for label in y])

            # 학습 및 검증 데이터 분리
            X_train, X_val, y_train, y_val = train_test_split(X_augmented, y_augmented, test_size=0.2, random_state=42)

            # 데이터를 Tensor로 변환
            X_train_tensor = torch.tensor(X_train, dtype=torch.float32).permute(0, 2, 1)  # (batch_size, features, time_steps)
            y_train_tensor = torch.tensor(y_train, dtype=torch.long)
            X_val_tensor = torch.tensor(X_val, dtype=torch.float32).permute(0, 2, 1)  # (batch_size, features, time_steps)
            y_val_tensor = torch.tensor(y_val, dtype=torch.long)

            # 새 클래스 수로 업데이트
            num_classes = len(unique_labels)

            # 모델 초기화
            input_shape = (X_train_tensor.size(1), X_train_tensor.size(2))  # (features, time_steps) for Conv1D + LSTM
            model = ConvLSTMEmotionClassifier(input_shape, num_classes).to(config['device'])

            # 손실 함수와 옵티마이저 정의
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

            # 학습 설정
            num_epochs = config['num_epochs']
            batch_size = config['batch_size']

            best_val_accuracy = 0.0
            best_epoch = 0
            best_train_loss = float('inf')
            best_val_loss = float('inf')
            best_train_accuracy = 0.0
            best_val_accuracy_epoch = 0
            best_preds_epoch = []
            best_labels_epoch = []

            # Train model
            for epoch in range(num_epochs):
                model.train()
                running_loss = 0.0
                total_correct = 0
                total_samples = 0

                for i in range(0, X_train_tensor.size(0), batch_size):
                    X_batch = X_train_tensor[i:i + batch_size].to(config['device'])
                    y_batch = y_train_tensor[i:i + batch_size].to(config['device'])

                    # Forward pass
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)

                    # Backward pass and optimization
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()

                    # Accuracy 계산
                    _, preds = torch.max(outputs, 1)
                    total_correct += (preds == y_batch).sum().item()
                    total_samples += y_batch.size(0)

                train_accuracy = total_correct / total_samples
                train_loss = running_loss / (X_train_tensor.size(0) // batch_size)
                print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}")

                # 검증 단계
                model.eval()
                val_loss = 0.0
                total_correct = 0
                total_samples = 0
                all_preds = []
                all_labels = []

                with torch.no_grad():
                    for i in range(0, X_val_tensor.size(0), batch_size):
                        X_batch = X_val_tensor[i:i + batch_size].to(config['device'])
                        y_batch = y_val_tensor[i:i + batch_size].to(config['device'])

                        outputs = model(X_batch)
                        loss = criterion(outputs, y_batch)
                        val_loss += loss.item()

                        # Accuracy 계산
                        _, preds = torch.max(outputs, 1)
                        total_correct += (preds == y_batch).sum().item()
                        total_samples += y_batch.size(0)

                        # 예측값 저장
                        all_preds.extend(preds.cpu().numpy())
                        all_labels.extend(y_batch.cpu().numpy())

                val_accuracy = total_correct / total_samples
                val_loss = val_loss / (X_val_tensor.size(0) // batch_size)
                print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

                # 최고 성능 저장
                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    best_epoch = epoch + 1
                    best_train_loss = train_loss
                    best_val_loss = val_loss
                    best_train_accuracy = train_accuracy
                    best_val_accuracy_epoch = val_accuracy
                    best_preds_epoch = all_preds
                    best_labels_epoch = all_labels

            # Update best overall accuracy
            if best_val_accuracy > best_overall_accuracy:
                best_overall_accuracy = best_val_accuracy
                best_combination = (noise_factor, shift_range)
                best_preds = best_preds_epoch
                best_labels = best_labels_epoch
                best_epoch_final = best_epoch

    # 최종 평가 결과 출력
    print(f"Best overall validation accuracy: {best_overall_accuracy:.4f} with noise_factor={best_combination[0]} and shift_range={best_combination[1]} at Epoch {best_epoch_final}")
    print("Classification Report (Best Epoch):")
    print(classification_report(best_labels, best_preds, target_names=[str(i) for i in range(num_classes)]))

    # # 혼동 행렬 시각화 및 저장 (Best Epoch에 대해서만)
    # plot_confusion_matrix(best_labels, best_preds, num_classes, save_path="confusion_matrix_best_combination.png")

if __name__ == "__main__":
    # GPU 또는 CPU 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main()
