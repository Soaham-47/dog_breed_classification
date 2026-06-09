import torch.nn as nn
import torchvision.models as models

def create_res_model(output_shape: int):
    """
    Builds the exact structural skeleton of the fine-tuned ResNet50 model
    matching the training pipeline for perfect weight file dictionary mapping.
    """
    # 1. Initialize the baseline pretrained weights skeleton
    model = models.resnet50(weights="DEFAULT")

    # 2. Freeze all pretrained layers
    for param in model.parameters():
        param.requires_grad = False

    # 3. Unfreeze only the last bottleneck block of layer4 
    # (Matches training setup, though ignored during final inference eval mode)
    for param in model.layer4[-1].parameters():
        param.requires_grad = True

    # 4. EXACT MATCH REPLACEMENT: Recreate your dropout + linear layer sequence
    model.fc = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(model.fc.in_features, output_shape)
    )

    # Return only the configured model skeleton (no optimizer/loss needed for deployment)
    return model, None, None