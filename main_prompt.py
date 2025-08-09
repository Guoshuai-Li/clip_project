def main():
    print("Loading dataset...")
    dataset = load_flickr30k_dataset(n=10)  # Moderate sample size for stable evaluation

    print("\nStarting Prompt Engineering retrieval evaluation...")
    prompt_results = evaluate_prompts_i2t(dataset, prompt_templates, top_k=5)

    print("\nComparison of retrieval metrics across prompt templates:")
    df_metrics = compare_prompt_metrics(prompt_results)
    print(df_metrics)

    # Save results for further visualization or analysis
    df_metrics.to_csv("prompt_comparison_metrics.csv", index=True)
    print("\nMetrics saved to CSV file: prompt_comparison_metrics.csv")
    
    plot_prompt_comparison_inline("prompt_comparison_metrics.csv")
    
if __name__ == "__main__":
    main()

