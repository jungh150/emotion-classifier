import pandas as pd
import librosa
import numpy as np

def load_data(csv_path):
    """CSV 파일에서 데이터를 로드하고 파일 경로와 라벨을 반환"""
    data = pd.read_csv(csv_path)
    file_paths = data['file_path'].values
    labels = data['emotion_label'].values
    return file_paths, labels

def extract_features(file_path, n_mfcc=40, duration=2.5, offset=0.5):
    """MFCC 특징 추출"""
    data, sampling_rate = librosa.load(file_path, duration=duration, offset=offset)
    mfccs = librosa.feature.mfcc(y=data, sr=sampling_rate, n_mfcc=n_mfcc)
    return np.mean(mfccs.T, axis=0)

def preprocess_data(file_paths, labels):
    """특징 추출 및 배열화"""
    features = [extract_features(path) for path in file_paths]
    features = np.array(features)

    # 입력 데이터를 3차원으로 변환 (batch_size, time_steps=1, features)
    features = np.expand_dims(features, axis=-1)
    return features, np.array(labels)
