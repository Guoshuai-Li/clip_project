import torch
import torch.nn.functional as F

def cosine_top_k(query_embed, target_embeds, target_list, k=5):
    """
    Generic top-k retrieval function using cosine similarity.
    Supports both image-to-text and text-to-image retrieval.

    Args:
        query_embed (Tensor): Query embedding tensor of shape [1, D].
        target_embeds (Tensor): Embeddings of target items, shape [N, D].
        target_list (List): Original list of targets (e.g., captions or images).
        k (int): Number of top matches to return. Default is 5.

    Returns:
        List[Tuple[item, float]]: List of top-k matched items and their similarity scores.
    """
    # Compute cosine similarity between query and all targets
    sims = F.cosine_similarity(query_embed, target_embeds)
    
    # Get the indices of the top-k most similar items
    topk = torch.topk(sims, k=k)
    
    # Return the top-k items and their similarity scores
    return [(target_list[i], sims[i].item()) for i in topk.indices]

def image_to_text(image_embed, text_embeds, captions, k=5):
    """
    Perform image-to-text retrieval using CLIP embeddings.

    Args:
        image_embed (Tensor): Embedding of the input image (shape [1, D]).
        text_embeds (Tensor): Embeddings of the candidate captions (shape [N, D]).
        captions (List[str]): Original list of text captions.
        k (int): Number of top matches to return. Default is 5.

    Returns:
        List[Tuple[str, float]]: Top-k captions with similarity scores.
    """
    return cosine_top_k(image_embed, text_embeds, captions, k)

def text_to_image(text_embed, image_embeds, images, k=5):
    """
    Perform text-to-image retrieval using CLIP embeddings.

    Args:
        text_embed (Tensor): Embedding of the input text (shape [1, D]).
        image_embeds (Tensor): Embeddings of the candidate images (shape [N, D]).
        images (List): Original list of image objects or references.
        k (int): Number of top matches to return. Default is 5.

    Returns:
        List[Tuple[Any, float]]: Top-k images with similarity scores.
    """
    return cosine_top_k(text_embed, image_embeds, images, k)
