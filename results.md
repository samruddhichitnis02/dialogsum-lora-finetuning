# Experimental Results

## Overview

This project adapted `google/flan-t5-small` for dialogue summarization using parameter-efficient LoRA fine-tuning. The models were evaluated on 499 held-out DialogSum examples using ROUGE-1, ROUGE-2, and ROUGE-L F1 scores.

All comparisons used the same test split and deterministic greedy decoding configuration:

```python
num_beams = 1
do_sample = False
```

## Base Model vs Fine-Tuned Model

The LoRA fine-tuned model substantially outperformed the original FLAN-T5-small base model.

| Metric | Base Model | Fine-Tuned Model | Improvement |
|---|---:|---:|---:|
| ROUGE-1 F1 | 0.0784 | 0.2625 | +0.1842 |
| ROUGE-2 F1 | 0.0146 | 0.0788 | +0.0642 |
| ROUGE-L F1 | 0.0689 | 0.2238 | +0.1549 |

The improvement indicates that fine-tuning helped the model learn the structure and vocabulary of the DialogSum summarization task.

## LoRA Rank Experiment

Two adapters were trained with the same base model and training configuration while varying the LoRA rank.

| Metric | LoRA `r=4` | LoRA `r=16` | Improvement |
|---|---:|---:|---:|
| ROUGE-1 F1 | 0.2625 | 0.3131 | +0.0506 |
| ROUGE-2 F1 | 0.0788 | 0.1027 | +0.0239 |
| ROUGE-L F1 | 0.2238 | 0.2599 | +0.0361 |

The `r=16` adapter achieved the best performance across all three metrics. The higher rank provided additional trainable capacity, allowing the adapter to capture more task-specific patterns.

## Training Behavior

### LoRA `r=4`

- Training epochs: 2
- Final training loss: approximately `14.95`
- The loss decreased consistently during training before reaching a stable range.

### LoRA `r=16`

- Training epochs: 2
- Final logged training loss: approximately `5.94`
- Training loss decreased from `10.27` at step 50 to `5.94` at step 1000.

The lower final training loss for `r=16` is consistent with its stronger held-out ROUGE scores in this experiment.

## Output Diversity and Error Analysis

The fine-tuned `r=4` model generated:

- 499 total predictions
- 166 unique summaries
- 33.27% unique-output ratio
- 45 examples containing repeated sentences
- 36 suspiciously short predictions
- 45 suspiciously long predictions

The most common failure patterns were:

- Incorrect speaker attribution
- Repeated phrases or sentences
- Missing the central event in a dialogue
- Grammatically incorrect or unnatural wording
- Overly literal summaries that captured only one sentence

The error analysis demonstrates that improved ROUGE scores do not guarantee perfect factual accuracy on every individual dialogue.

## Qualitative Comparison

The original base model sometimes produced generic responses that did not summarize the dialogue. The fine-tuned model generally recognized the dialogue topic and produced a more summary-like response. However, some predictions still contained speaker-role errors and repetition.

This suggests that the fine-tuned model learned useful task-specific behavior, but further improvements would be needed for production-quality summarization.

## Final Conclusion

LoRA fine-tuning successfully improved FLAN-T5-small for dialogue summarization while keeping the training process feasible on a 6 GB RTX 3060 GPU. The fine-tuned model improved ROUGE-1, ROUGE-2, and ROUGE-L over the base model, and the `r=16` adapter performed better than the `r=4` adapter.

The strongest configuration tested was:

```text
Base model: google/flan-t5-small
Adapter: LoRA with r=16
ROUGE-1 F1: 0.3131
ROUGE-2 F1: 0.1027
ROUGE-L F1: 0.2599
```

Future work should investigate beam search, repetition penalties, no-repeat n-gram constraints, validation-based checkpoint selection, longer training, and evaluation with the standard Hugging Face ROUGE implementation.

