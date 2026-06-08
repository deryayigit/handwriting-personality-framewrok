import torch.nn as nn
import timm


class ViTRegressor(nn.Module):
    def __init__(
        self,
        model_name: str,
        output_dim: int = 5,
        dropout: float = 0.1,
        pretrained: bool = True
    ):
        super().__init__()

        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0
        )

        feature_dim = self.backbone.num_features

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.backbone(x)
        outputs = self.head(features)
        return outputs


def freeze_backbone(model: ViTRegressor, freeze: bool = True) -> None:
    for parameter in model.backbone.parameters():
        parameter.requires_grad = not freeze