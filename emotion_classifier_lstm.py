import torch
import torch.nn as nn

class LSTMEmotionClassifier(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(LSTMEmotionClassifier, self).__init__()