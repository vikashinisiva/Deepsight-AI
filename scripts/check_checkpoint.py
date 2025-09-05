import torch

# Check what's in the checkpoint file
checkpoint = torch.load('weights/best_model.pth', map_location='cpu')
print('Keys in checkpoint:', list(checkpoint.keys()))

if isinstance(checkpoint, dict):
    if 'model_state_dict' in checkpoint:
        print('✅ Found model_state_dict')
        model_keys = list(checkpoint['model_state_dict'].keys())
        print(f'Model has {len(model_keys)} parameters')
        print('First 5 keys:', model_keys[:5])
        print('Last 5 keys:', model_keys[-5:])
    if 'accuracy' in checkpoint:
        print(f'Accuracy: {checkpoint["accuracy"]:.2f}%')
else:
    print('Checkpoint is not a dictionary')
