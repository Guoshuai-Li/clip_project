from datasets import load_dataset
import random

def load_random_subset(n=100, seed=20020312):
    """
    Load a random subset of n samples from the Flickr30k test split.

    Args:
        n : Number of samples to randomly select.
        seed: Random seed for reproducibility.

    Returns:
        Dataset: A Hugging Face Dataset object containing n randomly selected samples.
    """
    # Load the Flickr30k dataset, using the 'test' split
    dataset = load_dataset("flickr30k", split="test")

    # Set the random seed to ensure the same random selection each time
    random.seed(seed)

    # Randomly sample n unique indices from the dataset
    indices = random.sample(range(len(dataset)), n)

    # Select and return the subset of data corresponding to the sampled indices
    return dataset.select(indices)
