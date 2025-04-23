---
title: Initialization
weight: 3
bookToc: true
---

## Parameter initialization

Initialization before starting the gradient descent.

Avoid all parameters set to same value. (**symmetry breaking**)

Uniform distribution in the range {{<katex>}}[-\epsilon,\epsilon]{{</katex>}}

Zero-mean Gaussian {{<katex>}}\mathcal{N}(0,\epsilon^2){{</katex>}}

{{<button href="https://pytorch.org/docs/stable/nn.init.html">}}nn.init{{</button>}}
