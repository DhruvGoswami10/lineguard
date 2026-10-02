"""Grad-CAM, written by hand with PyTorch hooks (no extra library).

Idea (Selvaraju et al., 2017): ResNet-18's last block, layer4, outputs 512
feature maps on a 7x7 grid (for a 224x224 image), so it still knows *where*
things are. Grad-CAM asks how strongly each feature map pushes the "defect"
score up (the gradient of that score), averages the gradient into one weight
per map, and sums the maps with those weights. Positive areas are where the
network found evidence for "defect".
"""
import torch
import torch.nn.functional as F


class GradCAM:
    """Attach to a layer, call with an image batch of 1, then .remove() when done."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.activations = None
        self.gradients = None
        self._handle = target_layer.register_forward_hook(self._on_forward)

    def _on_forward(self, module, inputs, output):
        """Keep the layer's output, and ask autograd to hand us its gradient later."""
        self.activations = output
        if output.requires_grad:  # not the case inside torch.no_grad()
            output.register_hook(self._on_backward)

    def _on_backward(self, grad):
        self.gradients = grad

    def __call__(self, x: torch.Tensor, class_idx: int = 1):
        """Return (heatmap as a 7x7 numpy array scaled to [0, 1], logits)."""
        self.model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = self.model(x)
            logits[0, class_idx].backward()  # gradient of the chosen class score
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # one weight per feature map
        cam = (weights * self.activations).sum(dim=1)[0]          # weighted sum over maps
        cam = F.relu(cam)                                         # keep evidence *for* the class
        cam = cam / (cam.max() + 1e-8)                            # scale to [0, 1] for display
        return cam.detach().cpu().numpy(), logits.detach()

    def remove(self) -> None:
        """Detach the hook so the model behaves normally again."""
        self._handle.remove()
