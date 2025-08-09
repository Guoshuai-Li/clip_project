import os
import zipfile
import pandas as pd
import json
from datasets import Dataset
from typing import Optional
from tqdm import tqdm

def load_flickr30k_dataset(
    base_dir: str = r"D:\1-LGS\AI\CLIP",
    n: Optional[int] = None,
    seed: int = 20020312,
    shuffle: bool = True
) -> Dataset:
    """
    Load and prepare the Flickr30k dataset from local CSV and image ZIP.

    Args:
        base_dir (str): Path to the folder containing CSV and ZIP files.
        n (int, optional): If set, return only n random samples.
        seed (int): Random seed for reproducibility.
        shuffle (bool): Whether to shuffle before sampling.

    Returns:
        Dataset: A Hugging Face Dataset object (full or sampled).
    """
    # Path configuration
    csv_path = os.path.join(base_dir, "flickr_annotations_30k.csv")
    zip_path = os.path.join(base_dir, "flickr30k-images.zip")
    img_dir = os.path.join(base_dir, "flickr30k-images")

    # Extract images on first run only
    if not os.path.exists(img_dir):
        print("Unzipping the image...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(img_dir)
        print("Image decompression completed.")

    # Load annotations CSV
    df = pd.read_csv(csv_path)

    # Parse JSON-formatted columns
    df['caption'] = df['raw'].apply(json.loads)
    df['sentids'] = df['sentids'].apply(json.loads)

    # Build absolute image paths
    df['image'] = df['filename'].apply(lambda fn: os.path.join(img_dir, fn))

    # Keep relevant fields
    df = df[['image', 'caption', 'sentids', 'split', 'img_id', 'filename']]

    # Convert to a Hugging Face Dataset
    dataset = Dataset.from_pandas(df)

    # Optional shuffling and sampling
    if shuffle:
        dataset = dataset.shuffle(seed=seed)
    if n is not None:
        dataset = dataset.select(range(n))

    return dataset
