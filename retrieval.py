import torch
import torch.nn.functional as F

def cosine_top_k(query_embed, target_embeds, target_list, k=5):
    """
    General top-k similarity retrieval function (supports image-to-text and text-to-image).

    Args:
        query_embed (torch.Tensor): Single query vector (shape: [1, D]).
        target_embeds (torch.Tensor): Candidate vectors to search (shape: [N, D]).
        target_list (list): Original items (texts or images) corresponding to target_embeds.
        k (int): Number of top matches to return.

    Returns:
        list of (item, similarity_score): Top-k matches with similarity scores.
    """
    sims = F.cosine_similarity(query_embed, target_embeds)
    topk = torch.topk(sims, k=k)
    return [(target_list[i], sims[i].item()) for i in topk.indices]


def image_to_text(image_embed, text_embeds, captions, k=5):
    """
    Image-to-Text retrieval.

    Args:
        image_embed (torch.Tensor): Embedding vector of the query image.
        text_embeds (torch.Tensor): Embedding vectors of all candidate texts.
        captions (list[str]): List of original captions.
        k (int): Number of top matches to return.

    Returns:
        list of (caption, similarity_score): Top-k matching captions with similarity scores.
    """
    return cosine_top_k(image_embed, text_embeds, captions, k)


def text_to_image(text_embed, image_embeds, images, k=5):
    """
    Text-to-Image retrieval.

    Args:
        text_embed (torch.Tensor): Embedding vector of the query text.
        image_embeds (torch.Tensor): Embedding vectors of all candidate images.
        images (list): List of original image objects or paths.
        k (int): Number of top matches to return.

    Returns:
        list of (image, similarity_score): Top-k matching images with similarity scores.
    """
    return cosine_top_k(text_embed, image_embeds, images, k)
