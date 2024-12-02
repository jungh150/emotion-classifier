# **Emotion-Classifier**

Emotion-Classifier는 음성 데이터를 기반으로 감정을 분류하고, FLAME PyTorch를 사용하여 예측된 감정에 따라 3D 얼굴 표정을 생성하는 프로젝트입니다. 이 프로젝트는 다양한 음성 데이터셋을 활용하여 모델을 학습시키고, 감정 분류 및 시각화를 목표로 합니다.

---

## **프로젝트 구조**

```
emotion-classifier/
├── build/
│   └── bdist.win-amd64/
├── datasets/
│   ├── CREMA-D/
│   ├── RAVDESS/
│   ├── SAVEE/
│   ├── TESS/
│   ├── test/
│   └── emotion_melpath_dataset.csv
├── dist/
├── FLAME_PyTorch.egg-info/
├── models/
├── results/
├── src/
│   ├── emotion_classifier/
│   │   ├── model/
│   │   │   ├── emotion_classifier_all.py
│   │   │   ├── emotion_classifier_cnn.py
│   │   │   ├── emotion_classifier_convlstm.py
│   │   │   ├── emotion_classifier_crnn.py
│   │   │   ├── emotion_classifier_data_aug.py
│   │   │   ├── emotion_classifier_gru.py
│   │   │   ├── emotion_classifier_lstm.py
│   │   │   ├── emotion_classifier_rnn.py
│   │   │   ├── emotion_classifier_wav2.py
│   │   │   └── data_augmentation_test.py
│   │   ├── test.py
│   │   ├── train_multi.py
│   │   ├── train.py
│   │   └── utils.py
│   ├── flame_pytorch/
│   │   ├── build/
│   │   ├── dist/
│   │   ├── flame_pytorch/
│   │   │   └── ...
│   │   └── main.py
```

---

## **주요 기능**

1. **감정 분류**:
   - 음성 데이터를 기반으로 `ConvLSTM`, `CNN`, `RNN` 등 다양한 모델을 학습하여 감정을 분류합니다.
   - 주요 감정 레이블: `neutral`, `happy`, `sad`, `angry`, `fear`, `disgust`.

2. **3D 얼굴 시각화**:
   - FLAME PyTorch를 사용하여 예측된 감정에 따라 3D 얼굴 표정을 생성합니다.
   - 감정별 `expression_params`를 설정하여 표정을 조정할 수 있습니다.

3. **데이터셋**:
   - 공개된 Kaggle 데이터셋을 활용하여 학습합니다:
     - [CREMA-D 데이터셋](https://www.kaggle.com/datasets/dmitrybabko/speech-emotion-recognition-en)
     - [RAVDESS 데이터셋](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio)
     - [SAVEE 데이터셋](https://www.kaggle.com/datasets/ejlok1/surrey-audiovisual-expressed-emotion-savee)
     - [TESS 데이터셋](https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess)

---

## **실행**

### 1. **모델 학습**
모델을 학습하려면 아래 명령어를 실행하세요:
```bash
python src/emotion_classifier/train.py
```

### 2. **테스트 및 시각화**
테스트 데이터로 감정을 분류하고, 예측된 감정을 기반으로 3D 얼굴 표정을 생성하려면 아래 명령어를 실행하세요:
```bash
python src/emotion_classifier/test.py
```
이 명령어는 다음 작업을 수행합니다:
- 음성 데이터를 사용해 감정을 예측합니다.
- 예측된 감정을 기반으로 3D 얼굴 표정을 생성하고 화면에 표시합니다.

---

## **출처 및 참고**

- **FLAME PyTorch**: [https://github.com/soubhiksanyal/FLAME_PyTorch](https://github.com/soubhiksanyal/FLAME_PyTorch)

- **데이터셋**:
  - CREMA-D 데이터셋 [CREMA-D 데이터셋](https://www.kaggle.com/datasets/dmitrybabko/speech-emotion-recognition-en)
  - [RAVDESS 데이터셋](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio)
  - [SAVEE 데이터셋](https://www.kaggle.com/datasets/ejlok1/surrey-audiovisual-expressed-emotion-savee)
  - [TESS 데이터셋](https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess)