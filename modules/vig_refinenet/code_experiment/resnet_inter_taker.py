import torch
from torchvision.models import resnet50
from torchvision.models._utils import IntermediateLayerGetter

backbone = resnet50(weights="IMAGENET1K_V1")

return_layers = {
    "conv1": "feat0",
    "layer1": "feat1",
    "layer2": "feat2",
    "layer3": "feat3",
    "layer4": "feat4",
}

model = IntermediateLayerGetter(backbone, return_layers)

x = torch.randn(1, 3, 224, 224)
features = model(x)

print(features.keys())
# dict_keys(['feat2', 'feat3', 'feat4'])

for k, v in features.items():
    print(f"{k}: {v.shape}")