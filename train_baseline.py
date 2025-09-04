import os, torch, torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

data_dir = "crops_subset"; os.makedirs("weights", exist_ok=True)
batch, epochs = 32, 2

tfm = transforms.Compose([
  transforms.Resize((224,224)),
  transforms.ToTensor(),
  transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

ds = datasets.ImageFolder(data_dir, transform=tfm)
n = len(ds); n_val = max(1, int(0.2*n))
train_ds, val_ds = random_split(ds, [n-n_val, n_val])
tl = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=0)
vl = DataLoader(val_ds, batch_size=batch, shuffle=False, num_workers=0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
for p in m.features.parameters(): p.requires_grad = False
m.classifier[1] = nn.Linear(m.classifier[1].in_features, 2)
m = m.to(device)
opt = torch.optim.AdamW(m.classifier[1].parameters(), lr=1e-3)
crit = nn.CrossEntropyLoss()

print(f"Dataset: {len(ds)} images, Train: {len(train_ds)}, Val: {len(val_ds)}")
print(f"Classes: {ds.classes}")

best, best_path = 0.0, "weights/baseline.pth"
for e in range(epochs):
    m.train()
    train_loss = 0.0
    for x,y in tqdm(tl, desc=f"train e{e+1}"):
        x,y = x.to(device), y.to(device)
        opt.zero_grad(); loss = crit(m(x), y); loss.backward(); opt.step()
        train_loss += loss.item()

    m.eval(); correct=tot=0
    val_loss = 0.0
    with torch.no_grad():
        for x,y in vl:
            x,y = x.to(device), y.to(device)
            out = m(x)
            val_loss += crit(out, y).item()
            pr = out.argmax(1)
            correct += (pr==y).sum().item(); tot += y.numel()
    acc = correct/tot if tot else 0
    
    print(f"Epoch {e+1}: Train Loss: {train_loss/len(tl):.4f}, Val Loss: {val_loss/len(vl):.4f}, Val Acc: {acc:.4f}")
    
    if acc>best: best=acc; torch.save(m.state_dict(), best_path)

print(f"Best validation accuracy: {best:.4f}")
print(f"Model saved to: {best_path}")