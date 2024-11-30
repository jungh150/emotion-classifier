import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from motion_cl import CNNEmotionClassifier #코드 수정해야됨
from utils import load_data
import librosa
import numpy as np
import os

# 데이터 증강 함수 정의
def add_noise(audio, noise_level=0.005):
    noise = np.random.normal(0, noise_level, audio.shape)
    return audio + noise

def pitch_shift(audio, sr, n_steps=2):
    return librosa.effects.pitch_shift(y=audio, sr=sr, n_steps=n_steps)

def volume_scale(audio, scale=1.5):
    return audio * scale

def time_shift(audio, shift_max=0.2):
    shift = int(np.random.uniform(-shift_max, shift_max) * len(audio))
    return np.roll(audio, shift)

def augment_audio(audio, sr):
    augmentations = [
        lambda x: add_noise(x),
        lambda x: pitch_shift(x, sr, n_steps=np.random.randint(-5, 5)),
        lambda x: volume_scale(x, scale=np.random.uniform(0.5, 1.5)),
        lambda x: time_shift(x)
    ]
    aug_func = np.random.choice(augmentations)
    return aug_func(audio)

def preprocess_data_with_augmentation(file_paths, labels, sr=16000, fixed_length=16000):
    """Preprocess data and apply augmentations, ensuring all audio has the same length."""
    augmented_data = []
    augmented_labels = []

    for i, path in enumerate(file_paths):
        audio, _ = librosa.load(path, sr=sr)
        audio = librosa.util.fix_length(data=audio, size=fixed_length)  # 고정 길이로 패딩/자르기
        
        # Add original audio
        augmented_data.append(audio)
        augmented_labels.append(labels[i])

        # Add augmented data (3 times)
        for _ in range(3):
            augmented_audio = augment_audio(audio, sr)
            augmented_audio = librosa.util.fix_length(data=augmented_audio, size=fixed_length)  # 고정 길이로 패딩/자르기
            augmented_data.append(augmented_audio)
            augmented_labels.append(labels[i])

    return np.array(augmented_data), np.array(augmented_labels)


def calculate_accuracy(y_true, y_pred):
    """Accuracy 계산 함수"""
    correct = (y_true == y_pred).sum().item()
    total = len(y_true)
    return correct / total

def main():
    csv_path = '/home/smk11602/TTS/EmotionModel/accurtest/emotion4_dataset.csv' #코드 수정해야됨
    file_paths, labels = load_data(csv_path)
    X, y = preprocess_data_with_augmentation(file_paths, labels)

    # 학습 및 검증 데이터 분리
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"X_train shape: {X_train.shape}")

    # 데이터에 채널 축 추가 (3차원 변환)
    X_train = np.expand_dims(X_train, axis=1)  # (batch_size, 1, time_steps)
    X_val = np.expand_dims(X_val, axis=1)      # (batch_size, 1, time_steps)

    # 데이터를 Tensor로 변환
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)  # (batch_size, channels, time_steps)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)      # (batch_size, channels, time_steps)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)

    # 모델 초기화
    input_shape = (X_train_tensor.size(1), X_train_tensor.size(2))  # (channels, time_steps)
    num_classes = len(np.unique(y))
    model = CNNEmotionClassifier(input_shape, num_classes).to(device)

    # 손실 함수와 옵티마이저 정의
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 학습 설정
    num_epochs = 60
    batch_size = 32

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        total_correct = 0
        total_samples = 0

        for i in range(0, X_train_tensor.size(0), batch_size):
            X_batch = X_train_tensor[i:i + batch_size].to(device)
            y_batch = y_train_tensor[i:i + batch_size].to(device)

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
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / (X_train_tensor.size(0) // batch_size):.4f}, Accuracy: {train_accuracy:.4f}")

        # 검증 단계
        model.eval()
        val_loss = 0.0
        total_correct = 0
        total_samples = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for i in range(0, X_val_tensor.size(0), batch_size):
                X_batch = X_val_tensor[i:i + batch_size].to(device)
                y_batch = y_val_tensor[i:i + batch_size].to(device)

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
        print(f"Validation Loss: {val_loss / (X_val_tensor.size(0) // batch_size):.4f}, Validation Accuracy: {val_accuracy:.4f}")

    # 최종 평가 결과 출력
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=[str(i) for i in range(num_classes)]))

    # 저장 경로 확인 및 디렉터리 생성
    save_dir = "model"
    os.makedirs(save_dir, exist_ok=True)

    # 모델 저장
    save_path = os.path.join(save_dir, "emotion_cl_lstmcnn_model_4_aug.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved as {save_path}")

if __name__ == "__main__":
    # GPU 또는 CPU 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main()
