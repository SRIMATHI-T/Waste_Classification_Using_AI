import torch
import torch.nn as nn
from torchvision.models import resnet50

from src.models.cbam import CBAM

class ResNet50CBAM(nn.Module):
    def __init__(self, num_classes=6):
        super(ResNet50CBAM, self).__init__()
        # Load standard resnet50
        base_model = resnet50(weights=None)
        
        # Copy layers
        self.conv1 = base_model.conv1
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool
        
        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4
        
        # Add CBAM after the ResNet features
        self.cbam = CBAM(2048)
        
        self.avgpool = base_model.avgpool
        self.fc = nn.Linear(2048, num_classes)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.cbam(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x

def load_model(model_path, num_classes=6, device='cpu'):
    """
    Load the ResNet50-CBAM model from the given path.
    """
    model = ResNet50CBAM(num_classes=num_classes)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def predict(model, image_tensor, device='cpu'):
    """
    Perform inference on an image tensor.
    """
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        
    return predicted.item(), confidence.item(), probabilities[0].cpu().numpy()
