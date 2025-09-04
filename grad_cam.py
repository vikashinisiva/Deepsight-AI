# grad_cam.py
import cv2, torch, numpy as np
from torchvision import transforms

class GradCAM:
    """
    Minimal Grad-CAM for image classification. Works with EfficientNet-B0.
    target_layer: the conv layer to hook (e.g., model.features[-1])
    """
    def __init__(self, model, target_layer):
        self.model = model.eval()
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        def fwd_hook(module, inp, out):
            self.activations = out.detach()

        def bwd_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.fh = target_layer.register_forward_hook(fwd_hook)
        self.bh = target_layer.register_full_backward_hook(bwd_hook)

    def remove_hooks(self):
        self.fh.remove()
        self.bh.remove()

    @torch.no_grad()
    def _resize_norm(self, cam, size_hw):
        cam = np.maximum(cam, 0)
        cam = cam / (cam.max() + 1e-8)
        cam = cv2.resize(cam, (size_hw[1], size_hw[0]))
        return cam

    def __call__(self, input_tensor, class_idx=None):
        """
        input_tensor: (1,C,H,W), normalized
        class_idx: index to compute Grad-CAM for; if None, use predicted class
        returns: cam (H,W) in [0,1]
        """
        # Reset gradients and activations
        self.activations = None
        self.gradients = None
        
        # Ensure gradient computation
        input_tensor.requires_grad_(True)
        
        # Forward
        logits = self.model(input_tensor)
        if class_idx is None:
            class_idx = torch.softmax(logits, dim=1).argmax(1).item()

        # Backward for the target class
        self.model.zero_grad()
        score = logits[0, class_idx]
        score.backward(retain_graph=True)

        # Extract acts & grads
        acts = self.activations  # (B, C, H, W)
        grads = self.gradients   # (B, C, H, W)
        
        if grads is None or acts is None:
            print("Warning: No gradients captured, returning zero CAM")
            return np.zeros((224, 224))
            
        weights = grads.mean(dim=(2,3), keepdim=True)  # (B,C,1,1)

        cam = (weights * acts).sum(dim=1, keepdim=True)  # (B,1,H,W)
        cam = cam[0,0].cpu().numpy()  # (H,W)
        # Note: we return raw CAM; caller should resize to original image
        return cam

def overlay_cam_on_image(bgr_img, cam_01, alpha=0.35):
    """
    bgr_img: HxWx3 uint8 (OpenCV)
    cam_01:  HxW float in [0,1] (already resized to image size)
    returns: overlay BGR uint8
    """
    heatmap = (cam_01 * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # BGR
    overlay = cv2.addWeighted(heatmap, alpha, bgr_img, 1 - alpha, 0)
    return overlay

# Common transform matching your training
def make_infer_transform():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
