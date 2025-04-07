import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import resnet
from transformers import CLIPModel, CLIPTokenizer

class RGBBranch(nn.Module):
    """
    Generate Model Architecture
    """

    def __init__(self, arch, scene_classes=397):
        super(RGBBranch, self).__init__()

        # --------------------------------#
        #          Base Network           #
        # ------------------------------- #
        if arch == 'ResNet-18':
            # ResNet-18 Network
            base = resnet.resnet18(pretrained=True)
            # Size parameters for ResNet-18
            size_fc_RGB = 512
        elif arch == 'ResNet-50':
            # ResNet-50 Network
            base = resnet.resnet50(pretrained=True)
            # Size parameters for ResNet-50
            size_fc_RGB = 2048

        # --------------------------------#
        #           RGB Branch            #
        # ------------------------------- #
        # First initial block
        self.in_block = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, return_indices=True)
        )

        # Encoder
        self.encoder1 = base.layer1
        self.encoder2 = base.layer2
        self.encoder3 = base.layer3
        self.encoder4 = base.layer4

        # -------------------------------------#
        #            RGB Classifier            #
        # ------------------------------------ #
        self.dropout = nn.Dropout(0.3)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(size_fc_RGB, scene_classes)

        # Loss
        self.criterion = nn.CrossEntropyLoss()

        # Freeze all feature extractor parameters except the classifier
        modules_to_freeze = [self.in_block, self.encoder1, self.encoder2, self.encoder3, self.encoder4]
        for module in modules_to_freeze:
            for param in module.parameters():
                param.requires_grad = False

    def forward(self, x):
        """
        Network forward
        :param x: RGB Image
        :return: Scene recognition predictions
        """
        # --------------------------------#
        #           RGB Branch            #
        # ------------------------------- #
        x, pool_indices = self.in_block(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # -------------------------------------#
        #            RGB Classifier            #
        # ------------------------------------ #
        act = self.avgpool(e4)
        act = act.view(act.size(0), -1)
        act = self.dropout(act)
        act = self.fc(act)

        return act

    def get_features(self, x):
        """
        Extract features without classification
        :param x: RGB Image
        :return: Feature representation
        """
        # --------------------------------#
        #           RGB Branch            #
        # ------------------------------- #
        x, pool_indices = self.in_block(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Extract features
        features = self.avgpool(e4)
        features = features.view(features.size(0), -1)
        
        return features

    def loss(self, x, target):
        """
        Function to compute the loss
        :param x: Predictions obtained by the network
        :param target: Ground-truth scene recognition labels
        :return: Loss value
        """
        # Check inputs
        assert (x.shape[0] == target.shape[0])

        # Classification loss
        loss = self.criterion(x, target.long())

        return loss

class DynamicContextIntegration(nn.Module):
    """
    A model that dynamically integrates visual and textual features for scene recognition.
    This model uses CLIP for text embedding and a gate mechanism for feature fusion.
    """
    def __init__(self, backbone='ResNet-50', num_classes=397, pretrained=True):
        super(DynamicContextIntegration, self).__init__()
        
        # RGB feature extractor
        self.rgb_branch = RGBBranch(arch=backbone, scene_classes=num_classes)
        
        # CLIP model for text embeddings
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        
        # Freeze CLIP model parameters
        for param in self.clip.parameters():
            param.requires_grad = False
            
        # Feature dimensions
        self.image_feature_dim = 2048 if backbone == 'ResNet-50' else 512
        self.text_feature_dim = self.clip.config.projection_dim  # Usually 512 for CLIP
        self.combined_dim = self.image_feature_dim + self.text_feature_dim
        
        # Gate mechanism
        self.sigmoid = nn.Sigmoid()
        self.gate_layer = nn.Linear(self.combined_dim, self.text_feature_dim)
        
        # Context integration classifier
        self.classifier = nn.Linear(self.text_feature_dim, num_classes)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, images, tags):
        """
        Forward pass with both images and text tags
        :param images: Batch of images
        :param tags: Batch of text tags (strings)
        """
        # Get image features from RGB branch
        image_features = self.rgb_branch.get_features(images)
        
        # Process text with CLIP to get text embeddings
        text_inputs = self.tokenizer(tags, padding=True, return_tensors="pt").to(images.device)
        with torch.no_grad():
            text_features = self.clip.get_text_features(**text_inputs)
        
        # Squeeze dimensions if needed (먼저 squeeze 적용)
        image_features = image_features.squeeze()
        text_features = text_features.squeeze()
        
        # Normalize features (그 다음 normalize 적용)
        image_features = torch.nn.functional.normalize(image_features, dim=-1)
        text_features = torch.nn.functional.normalize(text_features, dim=-1)
        
        # Concatenate features for gate computation
        concat_features = torch.cat([image_features, text_features], dim=-1)
        
        # Apply gate mechanism
        context_gate = self.sigmoid(self.gate_layer(concat_features))
        
        # Original implementation: weighted feature addition (not concatenation)
        combined_features = torch.add(
            context_gate * image_features,
            (1 - context_gate) * text_features
        )
        
        # Final classification
        logits = self.classifier(combined_features)
        
        return logits
    
    def loss(self, x, target):
        """
        Function to compute the loss
        :param x: Predictions obtained by the network
        :param target: Ground-truth scene recognition labels
        :return: Loss value
        """
        return self.criterion(x, target.long())
