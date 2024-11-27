import torch
import torch.nn as nn

class CNNEmotionClassifier(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(CNNEmotionClassifier, self).__init__()