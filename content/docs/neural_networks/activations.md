---
title: Activation functions
weight: 5
bookToc: true
---

## Sigmoid

Sigmoid/Logistic
{{< katex display=true >}}
\sigma(z) = \frac{1}{1+\exp(-z)}
{{< /katex >}}
The derivative is given by
{{< katex display=true >}}
\sigma'(z) = \sigma(z)(1-\sigma(z))
{{< /katex >}}

## ReLU

Rectified Linear Unit (ReLU)
{{< katex display=true >}}
\text{ReLU}(z) = \max(z,0)
{{< /katex >}}

> Nair and Hinton, 2010

## pReLU

parameterized Rectified Linear Unit (pReLU)
{{< katex display=true >}}
\text{pReLU}(z) = \max(z,0) + \alpha \min(z,0)
{{< /katex >}}

> He et al., 2015

## Tanh

Hyperbolic tangent.
{{< katex display=true >}}
\text{tanh}(z) = \frac{1-\exp(-2z)}{1+\exp(-2z)}
{{< /katex >}}
The derivative is given by
{{< katex display=true >}}
\text{tanh}'(z)= 1- \text{tanh}^2(z)
{{< /katex >}}

## GeLU
Gaussian error Linear Unit/Smooth ReLU
{{< katex display=true >}}
\text{GeLU}(z) = z \Phi(z)
{{< /katex >}}
where {{< katex >}}\Phi(z){{< /katex >}} is the standard Gaussian cumulative distribution.

> Hendrycks and Gimpel, 2016

## Swish

{{< katex display=true >}}
\text{Swish}_{\beta}(z) = z \sigma(\beta z)
{{< /katex >}}
> Ramachandran et al. 2017

## GLU
Gated Liner Unit
{{< katex display=true >}}
\text{GLU}(z) = z \odot \sigma(wz+b)
{{< /katex >}}



## SwiGLU

Swish Gated Liner Unit
{{< katex display=true >}}
\text{Swish}(z) = z \odot \text{Swish}_{\beta}(wz+b)
{{< /katex >}}

{{% hint info %}}
[GLU Variants Improve Transformer](https://arxiv.org/pdf/2002.05202), Noam Shazeer, 2020.

🤷 *We offer no explanation as to why these architectures seem to work; we attribute their success, as all else, to divine benevolence.*
{{% /hint %}}



## Collateral

https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity

