import torch.nn as nn
import timm


class ViTClassifier(nn.Module):
    def __init__(self, model_name: str, num_classes: int, dropout: float = 0.1, pretrained: bool = True):
        super().__init__()

        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0
        )

        feature_dim = self.backbone.num_features

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        logits = self.head(features)
        return logits


def freeze_backbone(model: ViTClassifier, freeze: bool = True) -> None:
    for parameter in model.backbone.parameters():
        parameter.requires_grad = not freeze