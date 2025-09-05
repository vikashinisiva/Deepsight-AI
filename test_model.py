import torch
from torchvision import models
import torch.nn as nn

# Test loading our best model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'🧪 Testing model loading on {device}')

try:
    # Load model with same architecture as app
    model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
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

    # Load our trained weights
    checkpoint = torch.load('weights/best_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print('✅ Model loaded successfully!')
    print(f'📊 Model accuracy from checkpoint: {checkpoint["accuracy"]:.2f}%')
    print(f'🏗️  Architecture: EfficientNet-B3')
    print(f'🔧 Custom classifier: 1536 -> 512 -> 256 -> 2')

    # Quick inference test
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    with torch.no_grad():
        output = model(dummy_input)
        probs = torch.softmax(output, dim=1)
        print(f'🧠 Test inference successful!')
        print(f'📈 Output shape: {output.shape}')
        print(f'🎯 Sample prediction: Real={probs[0,0]:.3f}, Fake={probs[0,1]:.3f}')

except Exception as e:
    print(f'❌ Error: {e}')
