---
title: Normalization
weight: 4
bookToc: true
---



## Batch normalization

<img src="../img/batch.jpeg" alt="Batch normalization" width="400"/>

In batch normalization the mean and variance are computed across the mini-batch separately for each feature/hidden unit. For a mini-batch of size B
{{< katex display=true >}}
\mu_i = \frac{1}{B} \sum_{n=1}^{B} a_{ni}
{{< /katex >}}
{{< katex display=true >}}
\sigma_i^2 = \frac{1}{B} \sum_{n=1}^{B} (a_{ni}-\mu_i)^2
{{< /katex >}}
We normalize the pre-activations as follows.
{{< katex display=true >}}
\hat{a}_{ni} = \frac{a_{ni}-\mu_i}{\sqrt{\sigma_i^2+\delta}}
{{< /katex >}}
{{< katex display=true >}}
\tilde{a}_{ni} = \gamma_i \hat{a}_{ni} + \beta_i
{{< /katex >}}

{{<button href="https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html#torch.nn.BatchNorm1d">}}PyTorch{{</button>}}
```python
m = nn.BatchNorm1d(num_features)
```

## Layer normalization

<img src="../img/layer.jpeg" alt="Layer normalization" width="300"/>

In layer normalization the mean and variance are computed across the feature/hidden unit for each example seprately.
{{< katex display=true >}}
\mu_n = \frac{1}{M} \sum_{i=1}^{M} a_{ni}
{{< /katex >}}
{{< katex display=true >}}
\sigma_n^2 = \frac{1}{M} \sum_{i=1}^{M} (a_{ni}-\mu_i)^2
{{< /katex >}}
We normalize the pre-activations as follows.
{{< katex display=true >}}
\hat{a}_{ni} = \frac{a_{ni}-\mu_n}{\sqrt{\sigma_n^2+\delta}}
{{< /katex >}}
{{< katex display=true >}}
\tilde{a}_{ni} = \gamma_n \hat{a}_{ni} + \beta_n
{{< /katex >}}

{{<button href="https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html#torch.nn.LayerNorm">}}PyTorch{{</button>}}
```python
layer_norm = nn.LayerNorm(enormalized_shape)
```

## Collateral

https://pytorch.org/docs/stable/nn.html#normalization-layers
