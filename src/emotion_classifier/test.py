import torch
from src.emotion_classifier.utils import preprocess_data
from src.emotion_classifier.model.emotion_classifier_convlstm import ConvLSTMEmotionClassifier
from src.flame_pytorch.main import visualize_expression
import torch.nn.functional as F

# 감정별 expression 설정
expression_mappings = {
    "neutral": torch.zeros(1, 50, dtype=torch.float32).cuda(),
    "happy": torch.tensor([[2.0, 0.0, 0.0, 2.0, 2.0, 2.0, 2.0, 0.0, 0.0, 2.0] + [0.0]*40], dtype=torch.float32).cuda(),
    "sad": torch.tensor([[-1.0, -0.5, 0.0, 0.0, -2.0, 0.0, -2.0, -2.0, 2.0, 0.0] + [0.0]*40], dtype=torch.float32).cuda(),
    "angry": torch.tensor([[-0.8, 4.0, 0.0, 0.0, -5.0, 3.0, 0.7, 0.0, 0.0, -2.0] + [0.0]*40], dtype=torch.float32).cuda(),
    "fear": torch.tensor([[-2.0, 0.0, 0.0, 0.0, 5.0, 5.0, 0.0, 0.0, 0.0, 0.0] + [0.0]*40], dtype=torch.float32).cuda(),
    "disgust": torch.tensor([[-2.0, 2.0, 0.0, 2.0, -2.0, 0.0, -3.0, 0.0, 0.0, -2.0] + [0.0]*40], dtype=torch.float32).cuda(),
}

def predict_and_visualize(model, file_path, class_names, device):
    """
    단일 음성 파일에 대한 감정 예측 및 3D 얼굴 시각화
    """
    # 데이터 전처리
    feature, _ = preprocess_data([file_path], [0])  # 임의의 라벨로 전처리
    feature = torch.tensor(feature, dtype=torch.float32).permute(0, 2, 1).to(device)

    # 모델 예측
    model.eval()
    with torch.no_grad():
        output = model(feature)
        probs = F.softmax(output, dim=1)
        _, pred = torch.max(output, 1)
        pred_label = class_names[pred.item()]
    
    print(f"Predicted emotion label for '{file_path}': {pred_label}")
    print(f"Prediction probabilities: {probs.cpu().numpy()}")

    # 3D 얼굴 시각화
    expression_params = expression_mappings[pred_label]
    visualize_expression(expression_params)

def main():
    # 테스트 음성 파일 경로
    test_file_path = 'datasets/test/JK_f12.wav'

    # 데이터 로드를 통해 input_shape 추출
    feature, _ = preprocess_data([test_file_path], [0])  # 임의의 라벨로 전처리
    feature_tensor = torch.tensor(feature, dtype=torch.float32).permute(0, 2, 1)
    input_shape = (feature_tensor.size(1), feature_tensor.size(2))  # input_shape 계산

    # 모델 초기화
    class_names = ["neutral", "happy", "sad", "angry", "fear", "disgust"]
    num_classes = len(class_names)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ConvLSTMEmotionClassifier(input_shape, num_classes).to(device)
    
    # 학습된 가중치 로드
    model_path = 'models/convlstm_classifier.pth'
    model.load_state_dict(torch.load(model_path))
    print(f"Model loaded from {model_path}")

    # 테스트
    predict_and_visualize(model, test_file_path, class_names, device)

if __name__ == "__main__":
    main()
