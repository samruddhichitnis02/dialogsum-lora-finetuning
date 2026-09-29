# DialogSum LoRA Summarization

Parameter-efficient fine-tuning of **Google FLAN-T5-small** for dialogue summarization using **LoRA (Low-Rank Adaptation)** and the DialogSum dataset.

This project demonstrates how a compact sequence-to-sequence model can be adapted to a specialized summarization task on consumer hardware. The implementation includes baseline inference, LoRA fine-tuning, held-out evaluation, error analysis, and a rank comparison between `r=4` and `r=16` adapters.

## Project Highlights

- Fine-tuned `google/flan-t5-small` with PEFT LoRA adapters.
- Trained and evaluated locally on an NVIDIA RTX 3060 Laptop GPU with 6 GB VRAM.
- Used memory-safe, batch-wise generation for evaluation.
- Compared the original base model with the fine-tuned model on 499 held-out dialogues.
- Investigated output repetition, short summaries, long summaries, and low-scoring examples.
- Compared LoRA ranks `r=4` and `r=16` under the same training setup.

## Results

### Base Model vs Fine-Tuned Model

The LoRA fine-tuned model substantially improved over the original FLAN-T5-small model on the held-out test set.

| Metric | Base Model | Fine-Tuned Model | Improvement |
|---|---:|---:|---:|
| ROUGE-1 F1 | 0.0784 | 0.2625 | +0.1842 |
| ROUGE-2 F1 | 0.0146 | 0.0788 | +0.0642 |
| ROUGE-L F1 | 0.0689 | 0.2238 | +0.1549 |

### LoRA Rank Comparison

Increasing the LoRA rank from `r=4` to `r=16` improved performance across all reported metrics.

| Metric | LoRA `r=4` | LoRA `r=16` | Improvement |
|---|---:|---:|---:|
| ROUGE-1 F1 | 0.2625 | 0.3131 | +0.0506 |
| ROUGE-2 F1 | 0.0788 | 0.1027 | +0.0239 |
| ROUGE-L F1 | 0.2238 | 0.2599 | +0.0361 |

The `r=16` adapter achieved the strongest overall result in this experiment.

## Model and Training Configuration

| Component | Configuration |
|---|---|
| Base model | `google/flan-t5-small` |
| Fine-tuning method | LoRA with Hugging Face PEFT |
| Dataset | DialogSum |
| Training examples | 1,999 |
| Held-out test examples | 499 |
| Input length | 512 tokens |
| Target length | 128 tokens |
| Epochs | 2 |
| Learning rate | `1e-4` |
| Warm-up steps | 50 |
| Per-device batch size | 1 |
| Gradient accumulation steps | 4 |
| Precision | FP32 |
| LoRA target modules | T5 query and value projections (`q`, `v`) |
| LoRA dropout | 0.05 |
| LoRA alpha | 32 |

The stable `r=4` run finished with a final training loss of approximately `14.95`. The `r=16` run finished with a final logged training loss of approximately `5.94` under the same two-epoch training schedule.

## Why LoRA?

Full fine-tuning updates every parameter in the base model, which increases memory and storage requirements. LoRA freezes the original model and learns small low-rank update matrices in selected layers. This makes it practical to adapt FLAN-T5-small on a laptop GPU while keeping the original model reusable.

## Repository Contents

```text
.
├── dialogsum_lora_finetuning.ipynb
├── README.md
├── requirements.txt
├── fine_tuned_error_analysis.csv
└── results.md
```

The exact filenames may vary depending on the final notebook version. The notebook contains the complete experimental workflow and recorded outputs.

## Workflow

The project follows this pipeline:

1. Load and inspect the DialogSum data.
2. Convert dialogues and summaries into model-ready inputs.
3. Generate baseline summaries with the original FLAN-T5-small model.
4. Configure and attach LoRA adapters.
5. Tokenize the training split.
6. Fine-tune the model using a memory-conscious training configuration.
7. Save the trained LoRA adapters.
8. Generate summaries on the held-out test split.
9. Calculate ROUGE-1, ROUGE-2, and ROUGE-L F1 scores.
10. Compare the base model with the fine-tuned model.
11. Analyze repeated outputs and failure cases.
12. Compare LoRA rank `r=4` with `r=16`.

## Setup

Create a Python environment and install the dependencies:

```bash
pip install -r requirements.txt
```

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open the notebook and run the cells in order. The dataset and pretrained model are downloaded automatically through Hugging Face when needed.

## Reproducing the Experiment

The notebook is designed for a CUDA-enabled GPU, but it can also run on CPU with significantly longer execution time.

Recommended hardware for the recorded run:

- NVIDIA GPU with approximately 6 GB VRAM
- At least 16 GB system RAM
- Python 3.10 or later

The training configuration uses FP32 and a batch size of 1 to remain within a 6 GB GPU memory budget. Evaluation uses batch-wise generation and clears temporary GPU tensors after each batch.

## Evaluation Notes

The evaluation set contains 499 held-out dialogues. The reported scores use the same token-overlap and longest-common-subsequence F1 calculations throughout the base-model, `r=4`, and `r=16` comparisons.

The model was evaluated with deterministic greedy decoding:

```python
num_beams=1
do_sample=False
```

The results therefore measure the behavior of this specific decoding configuration.

## Error Analysis

The fine-tuned model showed meaningful improvement, but the outputs were not uniformly perfect. Analysis of the 499 predictions found:

- 166 unique generated summaries.
- Some repeated summaries across related dialogues.
- 45 examples containing repeated sentences.
- 36 unusually short predictions.
- 45 unusually long predictions.

The failure cases commonly involved incorrect speaker attribution, missing the main event, repeated phrases, and occasional grammatical or factual errors. These observations are useful for improving decoding, increasing training data, tuning the training schedule, or using a larger base model.

## Limitations and Future Work

Potential improvements include:

- Use beam search or constrained decoding to reduce repetitive generations.
- Tune `max_new_tokens`, repetition penalty, and no-repeat n-gram settings.
- Train for additional epochs with validation-based early stopping.
- Evaluate with the standard Hugging Face `evaluate` ROUGE implementation for comparison with published benchmarks.
- Test additional LoRA ranks and target-layer combinations.
- Compare against larger FLAN-T5 variants when more GPU memory is available.
- Add validation-loss tracking and checkpoint selection.

## Technical Stack

- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- Hugging Face PEFT
- Jupyter Notebook
- CUDA
- NumPy and pandas

