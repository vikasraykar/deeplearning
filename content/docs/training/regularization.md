---
title: Regularization
weight: 5
bookToc: true
---

## Dropout

Dropout is one of the most effective form of regularization that is widely used.

The core idea is to delete nodes from the network, including their connections, at random during training.

Dropout is applied to both hidden and input nodes, but not outputs. It is equivalent to setting the output of a dropped node to zero.

{{<button href="https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html#torch.nn.Dropout">}}torch.nn.Dropout{{</button>}}

## Early stopping

For good generalization training should be stopped at the point of smallest error with respect to the validation set.
