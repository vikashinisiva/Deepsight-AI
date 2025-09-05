import torch

print('Checking all model files:')
models = ['weights/baseline.pth', 'weights/baseline_improved.pth', 'weights/advanced_model.pth', 'weights/best_model.pth']

for model_path in models:
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        if isinstance(checkpoint, dict):
            print(f'{model_path}: dict with keys {list(checkpoint.keys())}')
            if 'accuracy' in checkpoint:
                print(f'  Accuracy: {checkpoint["accuracy"]:.2f}%')
            if 'model_name' in checkpoint:
                print(f'  Model: {checkpoint["model_name"]}')
        else:
            print(f'{model_path}: direct state_dict with {len(checkpoint)} parameters')
    except Exception as e:
        print(f'{model_path}: Error - {e}')
