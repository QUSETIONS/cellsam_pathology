import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models.convnext import ConvNeXt_Tiny_Weights

class ConvNeXtFeatureExtractor(nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
        
        # 1. Load ConvNeXt Tiny
        # We use 'DEFAULT' weights (ImageNet)
        print("Loading ConvNeXt Tiny (ImageNet weights)...")
        self.model = models.convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
        
        # 2. Modify Architecture for Feature Extraction
        # ConvNeXt's classifier is a Sequential:
        # (0): LayerNorm2d((768,), eps=1e-06, elementwise_affine=True)
        # (1): Flatten(start_dim=1, end_dim=-1)
        # (2): Linear(in_features=768, out_features=1000, bias=True)
        
        # We want the embedding before the final Linear layer.
        # Actually, usually we want the output of the final pooling/norm but before projection.
        # The 'avgpool' is global average pooling (AdaptiveAvgPool2d(1)).
        # The 'classifier' block in torchvision implementation handles flattening and the final FC.
        
        # Replacing the Linear layer (classifier[2]) with Identity 
        # keeps the LayerNorm and Flatten, which is good.
        self.model.classifier[2] = nn.Identity()
        
        self.model.to(self.device)
        self.model.eval()

        # 3. Output Dimension
        # ConvNeXt Tiny dim is 768
        self.output_dim = 768

    def get_transforms(self):
        """
        Returns the standard transforms expected by ConvNeXt.
        """
        return transforms.Compose([
            transforms.ToPILImage(), # If input is numpy
            transforms.Resize(236),  # Slightly larger than 224
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def extract(self, patch_tensor):
        """
        Extract features from a batch of patches.
        Args:
            patch_tensor: Tensor of shape (B, C, H, W) normalized.
        Returns:
            numpy array of shape (B, 768)
        """
        if patch_tensor.dim() == 3:
            patch_tensor = patch_tensor.unsqueeze(0)
            
        patch_tensor = patch_tensor.to(self.device)
        
        # Forward pass
        features = self.model(patch_tensor)
        
        return features.cpu().numpy()

# --- Integration Helper ---
def upgrade_pipeline_to_convnext(pipeline_instance):
    """
    Call this to hot-swap the extractor in your existing pipeline.
    """
    print("Upgrading feature extractor to ConvNeXt...")
    new_extractor = ConvNeXtFeatureExtractor()
    pipeline_instance.feature_extractor = new_extractor
    pipeline_instance.transforms = new_extractor.get_transforms()
    # Note: DB schema upgrade required if storing embeddings!
    print("WARNING: Embedding dimension changed from 512 (ResNet) to 768 (ConvNeXt).")
    print("Please ensure your database 'wsi_features.db' is recreated.")
