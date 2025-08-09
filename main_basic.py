from PIL import Image

def show_i2t_example(image, image_embed, text_embeds, captions, k=3):
    print("\n[Image → Text Retrieval Example]")
    results = image_to_text(image_embed, text_embeds, captions, k)
    for i, (cap, score) in enumerate(results):
        print(f"Top {i+1}: {cap}  (score: {score:.4f})")

def show_t2i_example(caption, text_embed, image_embeds, images, k=3):
    print("\n[Text → Image Retrieval Example]")
    results = text_to_image(text_embed, image_embeds, images, k)
    for i, (img, score) in enumerate(results):
        print(f"Top {i+1}:  similarity = {score:.4f}")

def main():
    print("Loading dataset...")
    dataset = load_flickr30k_dataset(n=5)

    images = [sample["image"] for sample in dataset]
    captions = [sample["caption"][0] for sample in dataset]

    print("Extracting image and text features...")
    image_embeds = get_image_features(images)
    text_embeds = get_text_features(captions)

    # Example output
    show_i2t_example(images[0], image_embeds[0].unsqueeze(0), text_embeds, captions)
    show_t2i_example(captions[0], text_embeds[0].unsqueeze(0), image_embeds, images)

    # Retrieval evaluation
    print("\nEvaluating Image → Text retrieval...")
    i2t_metrics = evaluate_retrieval(dataset, top_k=5)
    for k, v in i2t_metrics.items():
        print(f"{k}: {v:.4f}")

    print("\nEvaluating Text → Image retrieval...")
    t2i_metrics = evaluate_text_to_image(dataset, top_k=5)
    for k, v in t2i_metrics.items():
        print(f"{k}: {v:.4f}")

if __name__ == "__main__":
    main()

