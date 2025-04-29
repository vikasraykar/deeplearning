---
title: Alignment
weight: 3
bookToc: true
---

# Alignment

LLMs are typically trained for next-token prediction.

Pre-trained LLMs may not be able to follow user instructions because they were not trained to do so.

Pre-trained LLMs may generate harmful content or perpetuate  biases inherent in their training data.

{{<mermaid>}}
---
title: LLM training stages
---
flowchart LR
    subgraph Pre-training
    A[Pre-training]
    end
    subgraph Post-training
    B["Instruction Alignment (SFT)"]
    C["Preference Alignment (RLHF)"]
    end
    subgraph Inference
    D[Prompt engineering]
    end
    A-->B
    B-->C
    C-->D
{{</mermaid>}}

## Fine tune LLMs with labelled data

### Supervised Fine Tuning (SFT)

Training data is task specific instructions paired with their expected outputs.

During backward pass we the force the loss corresponding to the instruction to be zero.

### Parameter Efficient Fine Tuning (PEFT)

### LoRA

### QLoRA

### Soft prompts

## Fine tune LLMs with reward models

## Alignment during inference

Prompting
