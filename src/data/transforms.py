from torchvision import transforms
from PIL import Image

def get_inference_transforms():
    """
    Returns the standard transforms for inference.
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

def process_image(image_file):
    """
    Load an image file (or take a PIL Image directly) and apply the inference transforms.
    """
    if isinstance(image_file, Image.Image):
        image = image_file.convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
        
    transform = get_inference_transforms()
    # Add batch dimension
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor
