"""
Modified CustomRCNN with Attention Mechanisms and Enhanced Detection Head

This module provides an enhanced version of CustomRCNN (Faster R-CNN) with:
1. Channel Attention Module (CAM) in FPN layers
2. Spatial Attention Module (SAM) in FPN layers  
3. Enhanced ROI Head with additional FC layers and dropout
4. Better regularization with batch normalization

Author: [Your Name]
Date: October 2025
Course: CMPE 249 - Deep Learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torchvision.ops import MultiScaleRoIAlign
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork

# Import base CustomRCNN and its components
from DeepDataMiningLearning.detection.modeling_rpnfasterrcnn import CustomRCNN
from DeepDataMiningLearning.detection.backbone import MyBackboneWithFPN


# ==================== ATTENTION MODULES ====================

class ChannelAttentionModule(nn.Module):
    """
    Channel Attention Module (CAM) for feature enhancement.
    
    This module adaptively recalibrates channel-wise feature responses by
    explicitly modeling interdependencies between channels. It uses both
    average and max pooling for better feature selection.
    
    Reference: CBAM (Convolutional Block Attention Module)
    https://arxiv.org/abs/1807.06521
    
    Args:
        channels (int): Number of input channels
        reduction (int): Reduction ratio for the squeeze operation (default: 16)
    
    Shape:
        - Input: (B, C, H, W)
        - Output: (B, C, H, W) with channel-wise attention applied
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # Global average pooling
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # Global max pooling
        
        # Shared MLP (implemented as 1x1 convolutions)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """
        Forward pass with dual-path (avg + max) channel attention.
        
        Args:
            x (torch.Tensor): Input feature map (B, C, H, W)
            
        Returns:
            torch.Tensor: Attention-weighted feature map (B, C, H, W)
        """
        # Average pooling path
        avg_out = self.fc(self.avg_pool(x))  # (B, C, 1, 1)
        
        # Max pooling path  
        max_out = self.fc(self.max_pool(x))  # (B, C, 1, 1)
        
        # Combine and apply sigmoid
        attention = self.sigmoid(avg_out + max_out)  # (B, C, 1, 1)
        
        # Apply attention weights
        return x * attention


class SpatialAttentionModule(nn.Module):
    """
    Spatial Attention Module (SAM) for spatial feature enhancement.
    
    This module focuses on 'where' the important features are located in the
    spatial dimensions (H, W). It generates a spatial attention map by using
    the inter-spatial relationship of features.
    
    Reference: CBAM (Convolutional Block Attention Module)
    https://arxiv.org/abs/1807.06521
    
    Args:
        kernel_size (int): Kernel size for spatial convolution (default: 7)
    
    Shape:
        - Input: (B, C, H, W)
        - Output: (B, C, H, W) with spatial attention applied
    """
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7), 'Kernel size must be 3 or 7'
        padding = kernel_size // 2
        
        # Spatial attention convolution
        self.conv = nn.Conv2d(
            2, 1, kernel_size, 
            padding=padding, 
            bias=False
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """
        Forward pass with spatial attention.
        
        Args:
            x (torch.Tensor): Input feature map (B, C, H, W)
            
        Returns:
            torch.Tensor: Attention-weighted feature map (B, C, H, W)
        """
        # Aggregate channel information
        avg_out = torch.mean(x, dim=1, keepdim=True)  # (B, 1, H, W)
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # (B, 1, H, W)
        
        # Concatenate along channel dimension
        attention_map = torch.cat([avg_out, max_out], dim=1)  # (B, 2, H, W)
        
        # Generate spatial attention map
        attention = self.sigmoid(self.conv(attention_map))  # (B, 1, H, W)
        
        # Apply attention weights
        return x * attention


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM).
    
    Combines both channel and spatial attention sequentially.
    
    Args:
        channels (int): Number of input channels
        reduction (int): Reduction ratio for channel attention (default: 16)
        kernel_size (int): Kernel size for spatial attention (default: 7)
    """
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_attention = ChannelAttentionModule(channels, reduction)
        self.spatial_attention = SpatialAttentionModule(kernel_size)
    
    def forward(self, x):
        """Apply channel attention then spatial attention."""
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


# ==================== ENHANCED FPN WITH ATTENTION ====================

class EnhancedFPNWithAttention(FeaturePyramidNetwork):
    """
    Enhanced Feature Pyramid Network with attention mechanisms.
    
    Extends the standard FPN by adding CBAM (Channel + Spatial attention)
    to each pyramid level, improving multi-scale feature representation.
    
    Args:
        in_channels_list (list): List of input channels for each pyramid level
        out_channels (int): Number of output channels for all pyramid levels
        extra_blocks: Additional blocks (e.g., LastLevelMaxPool)
        use_attention (bool): Whether to use attention modules (default: True)
    """
    def __init__(self, in_channels_list, out_channels, extra_blocks=None, use_attention=True):
        super().__init__(in_channels_list, out_channels, extra_blocks)
        
        self.use_attention = use_attention
        
        if use_attention:
            # Add CBAM module for each pyramid level
            self.attention_modules = nn.ModuleList([
                CBAM(out_channels, reduction=16, kernel_size=7)
                for _ in range(len(in_channels_list))
            ])
            
            print(f"✓ Added CBAM attention to {len(in_channels_list)} FPN levels")
    
    def forward(self, x):
        """
        Forward pass with attention-enhanced FPN.
        
        Args:
            x (OrderedDict): Input feature maps from backbone
            
        Returns:
            OrderedDict: Enhanced multi-scale feature maps
        """
        # Standard FPN forward pass
        output = super().forward(x)
        
        # Apply attention to each level
        if self.use_attention:
            names = list(output.keys())
            for idx, name in enumerate(names):
                output[name] = self.attention_modules[idx](output[name])
        
        return output


# ==================== ENHANCED ROI HEAD ====================

class EnhancedFastRCNNPredictor(nn.Module):
    """
    Enhanced Fast R-CNN prediction head with additional layers and regularization.
    
    Replaces the standard single-layer predictor with a multi-layer network
    including batch normalization and dropout for better performance.
    
    Args:
        in_channels (int): Number of input features from ROI pooling
        num_classes (int): Number of classes (including background)
        hidden_dim (int): Hidden layer dimension (default: 1024)
        dropout_rate (float): Dropout probability (default: 0.5)
    
    Shape:
        - Input: (N, in_channels)
        - Output: 
            - cls_score: (N, num_classes)
            - bbox_pred: (N, num_classes * 4)
    """
    def __init__(self, in_channels, num_classes, hidden_dim=1024, dropout_rate=0.5):
        super().__init__()
        
        # Classification branch
        self.cls_fc1 = nn.Linear(in_channels, hidden_dim)
        self.cls_bn1 = nn.BatchNorm1d(hidden_dim)
        self.cls_dropout1 = nn.Dropout(dropout_rate)
        
        self.cls_fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.cls_bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.cls_dropout2 = nn.Dropout(dropout_rate * 0.6)
        
        self.cls_score = nn.Linear(hidden_dim // 2, num_classes)
        
        # Bounding box regression branch
        self.bbox_fc1 = nn.Linear(in_channels, hidden_dim)
        self.bbox_bn1 = nn.BatchNorm1d(hidden_dim)
        self.bbox_dropout1 = nn.Dropout(dropout_rate)
        
        self.bbox_pred = nn.Linear(hidden_dim, num_classes * 4)
        
        self.relu = nn.ReLU(inplace=True)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass of enhanced prediction head.
        
        Args:
            x (torch.Tensor): ROI features (N, in_channels)
            
        Returns:
            tuple: (classification scores, bbox predictions)
        """
        # Classification branch
        cls_feat = self.cls_fc1(x)
        cls_feat = self.cls_bn1(cls_feat)
        cls_feat = self.relu(cls_feat)
        cls_feat = self.cls_dropout1(cls_feat)
        
        cls_feat = self.cls_fc2(cls_feat)
        cls_feat = self.cls_bn2(cls_feat)
        cls_feat = self.relu(cls_feat)
        cls_feat = self.cls_dropout2(cls_feat)
        
        cls_score = self.cls_score(cls_feat)
        
        # Bounding box regression branch
        bbox_feat = self.bbox_fc1(x)
        bbox_feat = self.bbox_bn1(bbox_feat)
        bbox_feat = self.relu(bbox_feat)
        bbox_feat = self.bbox_dropout1(bbox_feat)
        
        bbox_pred = self.bbox_pred(bbox_feat)
        
        return cls_score, bbox_pred


# ==================== MODIFIED CUSTOM RCNN ====================

class ModifiedCustomRCNN(CustomRCNN):
    """
    Modified CustomRCNN with attention mechanisms and enhanced detection head.
    
    Enhancements over base CustomRCNN:
    1. Enhanced FPN with CBAM (Channel + Spatial) attention modules
    2. Improved ROI head with multi-layer FC, batch norm, and dropout
    3. Better regularization for improved generalization
    
    These modifications aim to improve feature learning and classification
    accuracy, especially for challenging object detection tasks like
    autonomous driving datasets (Waymo, nuScenes).
    
    Args:
        backbone_modulename (str): Name of backbone module (e.g., 'resnet50', 'resnet101')
        trainable_layers (int): Number of trainable backbone layers (default: 3)
        num_classes (int): Number of object classes including background (default: 91)
        out_channels (int): FPN output channels (default: 256)
        min_size (int): Minimum image size for preprocessing (default: 800)
        max_size (int): Maximum image size for preprocessing (default: 1333)
        use_attention (bool): Whether to use attention in FPN (default: True)
        enhanced_roi_head (bool): Whether to use enhanced ROI head (default: True)
    
    Example:
        >>> model = ModifiedCustomRCNN(
        ...     backbone_modulename='resnet101',
        ...     trainable_layers=5,
        ...     num_classes=4,  # 3 classes + background for Waymo
        ...     use_attention=True,
        ...     enhanced_roi_head=True
        ... )
        >>> model.eval()
        >>> # Training
        >>> images = [torch.rand(3, 800, 800)]
        >>> targets = [{'boxes': torch.tensor([[100, 100, 200, 200]]), 
        ...             'labels': torch.tensor([1])}]
        >>> losses = model(images, targets)
        >>> # Inference
        >>> predictions = model(images)
    """
    
    def __init__(
        self,
        backbone_modulename='resnet101',
        trainable_layers=5,
        num_classes=91,
        out_channels=256,
        min_size=800,
        max_size=1333,
        use_attention=True,
        enhanced_roi_head=True,
        **kwargs
    ):
        # Initialize base CustomRCNN (but we'll modify it)
        print("\n" + "="*70)
        print("CREATING MODIFIED CustomRCNN")
        print("="*70)
        print(f"Backbone: {backbone_modulename}")
        print(f"Trainable layers: {trainable_layers}")
        print(f"Number of classes: {num_classes}")
        print(f"Attention enabled: {use_attention}")
        print(f"Enhanced ROI head: {enhanced_roi_head}")
        
        # Store configuration
        self.use_attention = use_attention
        self.enhanced_roi_head = enhanced_roi_head
        
        # Initialize parent class
        super().__init__(
            backbone_modulename=backbone_modulename,
            trainable_layers=trainable_layers,
            num_classes=num_classes,
            out_channels=out_channels,
            min_size=min_size,
            max_size=max_size,
            **kwargs
        )
        
        # Apply modifications after initialization
        self._apply_modifications()
        
        print("="*70)
        print("✅ Modified CustomRCNN created successfully!")
        print("="*70 + "\n")
    
    def _apply_modifications(self):
        """Apply architectural modifications to the base model."""
        modifications_applied = []
        
        # Modification 1: Replace FPN with attention-enhanced version
        if self.use_attention and hasattr(self, 'backbone') and hasattr(self.backbone, 'fpn'):
            print("\n✓ Replacing FPN with attention-enhanced version...")
            
            old_fpn = self.backbone.fpn
            in_channels_list = old_fpn.in_channels_list if hasattr(old_fpn, 'in_channels_list') else None
            
            if in_channels_list:
                # Create new FPN with attention
                new_fpn = EnhancedFPNWithAttention(
                    in_channels_list=in_channels_list,
                    out_channels=old_fpn.inner_blocks[0].out_channels,
                    extra_blocks=old_fpn.extra_blocks if hasattr(old_fpn, 'extra_blocks') else None,
                    use_attention=True
                )
                
                # Copy weights from old FPN
                new_fpn.load_state_dict(old_fpn.state_dict(), strict=False)
                
                # Replace FPN
                self.backbone.fpn = new_fpn
                modifications_applied.append("Enhanced FPN with CBAM attention")
        
        # Modification 2: Replace ROI head predictor
        if self.enhanced_roi_head and hasattr(self, 'roi_heads'):
            print("✓ Replacing ROI head with enhanced multi-layer version...")
            
            # Get original predictor details
            old_predictor = self.roi_heads.box_predictor
            in_features = old_predictor.cls_score.in_features
            num_classes = old_predictor.cls_score.out_features
            
            # Create enhanced predictor
            new_predictor = EnhancedFastRCNNPredictor(
                in_channels=in_features,
                num_classes=num_classes,
                hidden_dim=1024,
                dropout_rate=0.5
            )
            
            # Replace predictor
            self.roi_heads.box_predictor = new_predictor
            modifications_applied.append("Enhanced ROI head with multi-layer FC + dropout")
        
        # Print summary
        if modifications_applied:
            print("\n📝 Modifications applied:")
            for idx, mod in enumerate(modifications_applied, 1):
                print(f"   {idx}. {mod}")
        else:
            print("\n⚠️  No modifications applied (features disabled or components not found)")


# ==================== HELPER FUNCTIONS ====================

def create_modified_customrcnn(
    backbone='resnet101',
    num_classes=91,
    trainable_layers=5,
    pretrained_backbone=True,
    use_attention=True,
    enhanced_roi_head=True,
    device='cuda'
):
    """
    Factory function to create a ModifiedCustomRCNN model.
    
    Args:
        backbone (str): Backbone architecture ('resnet50', 'resnet101', 'resnet152')
        num_classes (int): Number of classes including background
        trainable_layers (int): Number of trainable backbone layers
        pretrained_backbone (bool): Whether to use pretrained backbone weights
        use_attention (bool): Whether to add attention modules to FPN
        enhanced_roi_head (bool): Whether to use enhanced ROI head
        device (str): Device to place model on
    
    Returns:
        ModifiedCustomRCNN: Configured and initialized model
    
    Example:
        >>> model = create_modified_customrcnn(
        ...     backbone='resnet101',
        ...     num_classes=4,
        ...     trainable_layers=5,
        ...     device='cuda'
        ... )
    """
    model = ModifiedCustomRCNN(
        backbone_modulename=backbone,
        trainable_layers=trainable_layers,
        num_classes=num_classes,
        use_attention=use_attention,
        enhanced_roi_head=enhanced_roi_head
    )
    
    model.to(device)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n📊 Model Statistics:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Frozen parameters: {total_params - trainable_params:,}")
    print(f"   Trainable percentage: {100 * trainable_params / total_params:.2f}%")
    
    return model


if __name__ == "__main__":
    # Test the modified model
    print("Testing ModifiedCustomRCNN...")
    
    model = create_modified_customrcnn(
        backbone='resnet50',
        num_classes=4,
        trainable_layers=3,
        device='cpu'
    )
    
    # Test forward pass
    print("\nTesting forward pass...")
    images = [torch.rand(3, 800, 800)]
    targets = [{
        'boxes': torch.tensor([[100, 100, 200, 200]], dtype=torch.float32),
        'labels': torch.tensor([1], dtype=torch.int64)
    }]
    
    model.train()
    losses = model(images, targets)
    print(f"Training losses: {losses}")
    
    model.eval()
    with torch.no_grad():
        predictions = model(images)
    print(f"Prediction boxes shape: {predictions[0]['boxes'].shape}")
    
    print("\n✅ All tests passed!")
