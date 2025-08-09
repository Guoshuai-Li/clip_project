from PIL import Image
import random

def get_random_captions(dataset, n, exclude_caption):
    """
    Retrieve n random captions from the dataset excluding the ground truth.

    Args:
        dataset: Dataset containing image-caption pairs.
        n (int): Number of random captions to retrieve.
        exclude_caption (str): The ground truth caption to exclude.

    Returns:
        list[str]: List of randomly selected captions.
    """
    all_captions = [sample["caption"][0] for sample in dataset if sample["caption"][0] != exclude_caption]
    return random.sample(all_captions, n)


def evaluate_retrieval(dataset, top_k=5):
    """
    Evaluate image-to-text retrieval performance using Top-1 Accuracy and Recall@k.

    Args:
        dataset: Dataset sampled using load_random_subset().
        top_k (int): Number of top matches to consider.

    Returns:
        dict: {"Top-1 Accuracy": float, "Recall@k": float}
    """
    total = 0
    top1_hits = 0
    recall_k_hits = 0

    for sample in dataset:
        image_path = sample["image"]
        gt_caption = sample["caption"][0]

        # Prepare candidate captions: ground truth + (n-1) random distractors
        candidates = [gt_caption] + get_random_captions(dataset, n=4, exclude_caption=gt_caption)
        random.shuffle(candidates)

        # Extract embeddings
        image_embed = get_image_features([image_path])[0]
        text_embeds = get_text_features(candidates)

        # Perform retrieval
        results = image_to_text(image_embed, text_embeds, candidates, k=top_k)
        top_captions = [r[0] for r in results]

        # Evaluate correctness
        if top_captions[0] == gt_caption:
            top1_hits += 1
        if gt_caption in top_captions:
            recall_k_hits += 1

        total += 1

    return {
        "Top-1 Accuracy": top1_hits / total,
        f"Recall@{top_k}": recall_k_hits / total
    }


def get_random_images(dataset, n, exclude_image):
    """
    Retrieve n random image paths from the dataset excluding the ground truth.

    Args:
        dataset: Dataset containing image-caption pairs.
        n (int): Number of random images to retrieve.
        exclude_image (str): The ground truth image path to exclude.

    Returns:
        list[str]: List of randomly selected image paths.
    """
    all_images = [sample["image"] for sample in dataset if sample["image"] != exclude_image]
    return random.sample(all_images, n)


def evaluate_text_to_image(dataset, top_k=5):
    """
    Evaluate text-to-image retrieval performance using Top-1 Accuracy and Recall@k.

    Args:
        dataset: Dataset sampled using load_random_subset().
        top_k (int): Number of top matches to consider.

    Returns:
        dict: {"Top-1 Accuracy (Text→Image)": float, "Recall@k (Text→Image)": float}
    """
    total = 0
    top1_hits = 0
    recall_k_hits = 0

    for sample in dataset:
        caption = sample["caption"][0]
        gt_image = sample["image"]

        # Prepare candidate images: ground truth + (n-1) random distractors
        candidates = [gt_image] + get_random_images(dataset, n=4, exclude_image=gt_image)
        random.shuffle(candidates)

        # Extract embeddings
        text_embed = get_text_features([caption])
        image_embeds = get_image_features(candidates)

        # Perform retrieval
        results = text_to_image(text_embed, image_embeds, candidates, k=top_k)
        top_images = [r[0] for r in results]

        # Evaluate correctness
        if top_images[0] == gt_image:
            top1_hits += 1
        if gt_image in top_images:
            recall_k_hits += 1

        total += 1

    return {
        "Top-1 Accuracy (Text→Image)": top1_hits / total,
        f"Recall@{top_k} (Text→Image)": recall_k_hits / total
    }

