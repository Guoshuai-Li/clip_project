# CLIP Image-Text Retrieval with Prompt Engineering

## Overview
This project implements an **image-text retrieval system** using [OpenAI's CLIP](https://openai.com/research/clip) model, evaluated on the Flickr30k dataset.  
It includes:
- Basic image-to-text and text-to-image retrieval
- Evaluation metrics (Top-1 Accuracy, Recall@k)
- Prompt Engineering experiments to explore the effect of different text templates
- Visualization of retrieval performance across prompts

## Project Structure
```bash
clip_project/
├── README.md              # Project description & results
├── requirements.txt       # Python dependencies
├── data_loader.py         # Load and preprocess Flickr30k data
├── clip_model.py          # Load CLIP and extract embeddings
├── retrieval.py           # Similarity calculation & retrieval functions
├── evaluation.py          # Evaluation functions & sample output
├── prompt_engineering.py  # Prompt templates and evaluation
├── main_basic.py          # Entry point for basic retrieval
└── main_prompt.py         # Entry point for prompt engineering experiments
```

---

## Environment & Dependencies
This project runs in a **Python environment** with the following main dependencies:
- Pillow  
- datasets  
- matplotlib  
- pandas  
- torch  
- tqdm  
- transformers  

Please ensure these packages are installed before running the code.

---

## Dataset
The project uses the [Flickr30k dataset (Hugging Face version)](https://huggingface.co/datasets/lmms-lab/flickr30k), which contains images paired with multiple English captions.  
> Dataset example fields:  
> - **image**: image file  
> - **caption**: five textual descriptions per image  
> - **img_id**: image ID  
> - **filename**: file name  

### Obtaining the dataset
- Download from the Hugging Face dataset page or load using the `datasets` library
- Alternatively, obtain the original data from the official Flickr30k source

> **Copyright Notice**  
> The images in this dataset are **not owned** by us and are provided solely for **non-commercial research and educational purposes**.  
> Please **do not redistribute** the images.  
> If you use this dataset, please cite the following paper:
> ```
> P. Young, A. Lai, M. Hodosh, and J. Hockenmaier.
> From image description to visual denotations: New similarity metrics for semantic
> inference over event descriptions.
> Transactions of the Association for Computational Linguistics (to appear).
> ```

---

## How to Use

### 1. Basic Retrieval  
Running `main_basic.py` will:
- Load a small subset of Flickr30k
- Extract image and text embeddings
- Show top-k retrieval examples
- Report Top-1 Accuracy and Recall@k

### 2. Prompt Engineering  
Running `main_prompt.py` will:
- Apply multiple prompt templates to captions
- Evaluate retrieval performance for each template
- Save results to `prompt_comparison_metrics.csv`
- Optionally visualize results with bar charts

---

## Example Results

### Image → Text Retrieval
1. *"A group of individuals are standing in the middle of a room while others are sitting around them and watching."* (score: 0.2322)  
2. *"A girl holds her iced drink in her gloved-hand with a look of concentration on her face."* (score: 0.1372)  
3. *"Three boys with no shirts on are sitting on cement steps by the water."* (score: 0.0997)  

---

### Evaluation Metrics
| Task          | Top-1 Accuracy | Recall@5 |
|---------------|---------------:|---------:|
| Image → Text  | 1.0000          | 1.0000   |
| Text → Image  | 1.0000          | 1.0000   |

---

### Prompt Engineering Metrics
| Prompt Template                  | Top-1 Accuracy | Recall@5 |
|----------------------------------|---------------:|---------:|
| A photo of {}.                   | 1.0000          | 1.0000   |
| This is a picture showing {}.    | 1.0000          | 1.0000   |
| There is {} in the image.        | 1.0000          | 1.0000   |
| An image that depicts {}.        | 1.0000          | 1.0000   |
| A visual scene of {}.            | 1.0000          | 1.0000   |

---

## Important Notes on Results
- All experiments were run on a **very small subset** of the Flickr30k dataset  
- This made the retrieval task trivial, resulting in **perfect scores** (Top-1 Accuracy = 1.0, Recall@5 = 1.0)  
- These results **do not** represent realistic model performance on the full dataset  
- For meaningful evaluation, run the code on a **larger dataset split** with adequate GPU resources
