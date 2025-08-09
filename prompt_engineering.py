def apply_prompt_templates(captions, templates):
    """
    Apply multiple prompt templates to each original caption.

    Args:
        captions (list[str]): Original descriptive sentences.
        templates (list[str]): Prompt templates, e.g., "A photo of {}".

    Returns:
        dict: {template_str: [caption1_with_template, caption2_with_template, ...]}
    """
    prompt_variants = {}
    for template in templates:
        prompt_variants[template] = [template.format(c) for c in captions]
    return prompt_variants


# Optional prompt templates (extend as needed)
prompt_templates = [
    "A photo of {}.",
    "This is a picture showing {}.",
    "There is {} in the image.",
    "An image that depicts {}.",
    "A visual scene of {}.",
]
from tqdm import tqdm

def evaluate_prompts_i2t(dataset, prompt_templates, top_k=5):
    """
    Evaluate Image → Text retrieval for each prompt template.

    Args:
        dataset: A subset of the Flickr30k dataset.
        prompt_templates (list[str]): Multiple prompt templates used to wrap captions.
        top_k (int): Top-k value used for Recall calculation.

    Returns:
        dict: {template_str: {'Top-1 Accuracy': float, 'Recall@k': float}}
    """
    results = {}

    # Original captions
    base_captions = [sample["caption"][0] for sample in dataset]
    image_paths = [sample["image"] for sample in dataset]

    # Extract image embeddings (only once)
    print("🖼️ Extracting image embeddings...")
    image_embeds = get_image_features(image_paths)

    for template in tqdm(prompt_templates, desc="🔁 Evaluating different prompts"):
        # Apply template to generate new captions
        prompted_captions = [template.format(c) for c in base_captions]

        # Get corresponding text embeddings
        text_embeds = get_text_features(prompted_captions)

        # Retrieval evaluation
        total = 0
        top1_hits = 0
        recall_k_hits = 0

        for i in range(len(dataset)):
            query_img_embed = image_embeds[i].unsqueeze(0)
            query_caption = prompted_captions[i]

            # Construct distractors: ground truth + random captions (with same template)
            distractors = [
                template.format(c) for j, c in enumerate(base_captions) if j != i
            ]
            candidates = [query_caption] + random.sample(distractors, k=4)
            random.shuffle(candidates)

            candidate_text_embeds = get_text_features(candidates)
            results_list = image_to_text(query_img_embed, candidate_text_embeds, candidates, k=top_k)
            top_captions = [r[0] for r in results_list]

            if query_caption == top_captions[0]:
                top1_hits += 1
            if query_caption in top_captions:
                recall_k_hits += 1

            total += 1

        results[template] = {
            "Top-1 Accuracy": top1_hits / total,
            f"Recall@{top_k}": recall_k_hits / total
        }

    return results
import pandas as pd

def compare_prompt_metrics(prompt_results):
    """
    Organize metrics for multiple prompt templates into a DataFrame
    for easier visualization and comparison.

    Args:
        prompt_results (dict): Mapping of prompt template to evaluation metrics.

    Returns:
        pd.DataFrame: Metrics as rows, prompt templates as columns.
    """
    df = pd.DataFrame(prompt_results).T  # templates as rows
    df = df.rename_axis("Prompt Template").reset_index()
    return df.set_index("Prompt Template").T  # metrics as rows for easier comparison
import pandas as pd
import matplotlib.pyplot as plt

def plot_prompt_comparison_inline(csv_path):
    """
    Read saved prompt evaluation results and display bar charts in a notebook.

    Args:
        csv_path (str): Path to the CSV file.
    """
    # Load metrics
    df = pd.read_csv(csv_path, index_col=0)
    df = df.T  # rows: prompt templates, columns: metrics

    # Global plotting style (optional)
    plt.style.use("default")
    plt.rcParams.update({
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10
    })

    # Plot Top-1 Accuracy
    plt.figure(figsize=(10, 4))
    df["Top-1 Accuracy"].plot(kind="bar", color="skyblue")
    plt.title("Top-1 Accuracy Across Prompt Templates")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.05)
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", linestyle="--")
    plt.tight_layout()
    plt.show()

    # Plot Recall@5
    plt.figure(figsize=(10, 4))
    df["Recall@5"].plot(kind="bar", color="orange")
    plt.title("Recall@5 Across Prompt Templates")
    plt.ylabel("Recall")
    plt.ylim(0, 1.05)
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", linestyle="--")
    plt.tight_layout()
    plt.show()

