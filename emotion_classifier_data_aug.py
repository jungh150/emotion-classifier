import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from utils import load_data
import librosa
import numpy as np
import os

from emotion_classifier_convlstm import ConvLSTMEmotionClassifier

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

def train_model(model, X_train, y_train, X_val, y_val, config):
    # 학습 설정
    num_epochs = config['num_epochs']
    batch_size = config['batch_size']
    learning_rate = config['learning_rate']
    device = config['device']
    
    # 손실 함수와 옵티마이저 정의
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 학습 루프
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        # 배치 단위 학습
        for i in range(0, X_train.size(0), batch_size):
            X_batch = X_train[i:i+batch_size].to(device)
            y_batch = y_train[i:i+batch_size].to(device)
            
            # Forward 계산
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            # Backward 및 가중치 업데이트
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            total_correct += (preds == y_batch).sum().item()
            total_samples += y_batch.size(0)
        
        # 학습 정확도 출력
        train_accuracy = total_correct / total_samples
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / (X_train.size(0) // batch_size):.4f}, Accuracy: {train_accuracy:.4f}")
        
        # 검증 단계
        model.eval()
        val_loss = 0.0
        total_correct = 0
        total_samples = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for i in range(0, X_val.size(0), batch_size):
                X_batch = X_val[i:i+batch_size].to(device)
                y_batch = y_val[i:i+batch_size].to(device)
                
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
                
                _, preds = torch.max(outputs, 1)
                total_correct += (preds == y_batch).sum().item()
                total_samples += y_batch.size(0)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())
        
        # 검증 정확도 출력
        val_accuracy = total_correct / total_samples
        print(f"Validation Loss: {val_loss / (X_val.size(0) // batch_size):.4f}, Validation Accuracy: {val_accuracy:.4f}")
    
    return all_preds, all_labels

def main():
    # 데이터 로드 및 증강 적용
    csv_path = 'datasets/emotion_melpath_dataset.csv'
    file_paths, labels = load_data(csv_path)
    X, y = preprocess_data_with_augmentation(file_paths, labels)

    # 학습 및 검증 데이터 분리
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 데이터에 채널 축 추가 (3차원 변환)
    X_train = np.expand_dims(X_train, axis=1)  # (batch_size, 1, time_steps)
    X_val = np.expand_dims(X_val, axis=1)      # (batch_size, 1, time_steps)

    # 데이터를 Tensor 형태로 변환
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)

    # 모델 초기화
    input_shape = (X_train_tensor.size(1), X_train_tensor.size(2))
    num_classes = len(np.unique(y))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ConvLSTMEmotionClassifier(input_shape, num_classes).to(device)

    # 학습 설정
    config = {
        'num_epochs': 60,
        'batch_size': 32,
        'learning_rate': 0.001,
        'device': device
    }
    
    # 모델 학습
    all_preds, all_labels = train_model(model, X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, config)
    
    # 분류 보고서 출력
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=[str(i) for i in range(num_classes)]))
    
    # 모델 저장
    save_dir = 'models'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'convlstm_classifier_aug.pth')
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    main()
