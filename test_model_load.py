import torch
from torchvision import models
import torch.nn as nn

# Test model loading
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
m = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)
for p in m.features.parameters(): p.requires_grad = False

num_features = m.classifier[1].in_features
m.classifier = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(num_features, 512),
    nn.BatchNorm1d(512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, 256),
    nn.BatchNorm1d(256),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(256, 2)
)

checkpoint = torch.load('weights/best_model.pth', map_location=device)
m.load_state_dict(checkpoint['model_state_dict'])
print('Model loaded successfully')
print(f'Accuracy: {checkpoint.get("accuracy", "N/A")}')
