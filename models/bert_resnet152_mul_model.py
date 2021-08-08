import torch.nn as nn
import torchvision
from transformers import BertModel


class VNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        model = torchvision.models.resnet152(pretrained=pretrained)
        model.fc = nn.Linear(2048, 768)
        self.backbone = model

    def forward(self, images):
        return self.backbone(images)


class VQANet(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.v_net = VNet()
        self.q_net = BertModel.from_pretrained("bert-base-uncased")
        self.classifier = nn.Sequential(
            nn.Linear(768, 1024),
            nn.ReLU(True),
            nn.Linear(1024, num_classes)
        )

    def forward(self, v, q):
        v_feat = self.v_net(v)
        q_feat = self.q_net(**q)[1]
        out = self.classifier(v_feat * q_feat)
        return out
