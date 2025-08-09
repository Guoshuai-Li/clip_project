from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# Load the CLIP model and its processor
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

def get_image_features(image_paths):
    """
    Args:
        image_paths (list[str]): List of file paths to images.
    Returns:
        torch.Tensor: Feature vectors (CLIP embeddings) for the images.
    """
    images = [Image.open(p).convert("RGB") for p in image_paths]
    inputs = processor(images=images, return_tensors="pt")
    return model.get_image_features(**inputs)

def get_text_features(texts):
    """
    Args:
        texts (list[str]): List of text strings.
    Returns:
        torch.Tensor: Feature vectors (CLIP embeddings) for the texts.
    """
    inputs = processor(text=texts, return_tensors="pt", padding=True)
    return model.get_text_features(**inputs)
