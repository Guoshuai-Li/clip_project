from transformers import CLIPProcessor, CLIPModel

# Load the pretrained CLIP model and its processor from Hugging Face
# This version uses the ViT-L/14 architecture
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

def get_image_features(image):
    """
    Extract image features using the CLIP model.

    Args:
        image (PIL.Image): An input image in PIL format.

    Returns:
        torch.Tensor: A tensor containing the image embedding from the CLIP model.
    """
    # Preprocess the image into a format the model can understand
    inputs = processor(images=image, return_tensors="pt")
    
    # Compute and return the image features (embeddings)
    return model.get_image_features(**inputs)

def get_text_features(texts):
    """
    Extract text features using the CLIP model.

    Args:
        texts (List[str]): A list of input strings (sentences or phrases).

    Returns:
        torch.Tensor: A tensor containing the text embeddings from the CLIP model.
    """
    # Preprocess the input texts (tokenization, padding, etc.)
    inputs = processor(text=texts, return_tensors="pt", padding=True)
    
    # Compute and return the text features (embeddings)
    return model.get_text_features(**inputs)
