---
title: Linear Regression
weight: 1
bookToc: true
---

## Linear Regression

Linear Regression is a single layer neural network for regression.

{{<mermaid>}}
stateDiagram-v2
    direction LR
    z1: $$x_1$$
    z2: $$x_2$$
    zi: $$x_i$$
    zM: $$x_d$$
    aj: $$a=\sum_i w_{i} x_i$$
    zj: $$z=a$$
    z1 --> aj:$$w_{1}$$
    z2 --> aj:$$w_{2}$$
    zi --> aj:$$w_{i}$$
    zM --> aj:$$w_{d}$$
    aj --> zj
    zj --> END:::hidden
    note left of zM : Inputs
    note left of aj : Pre-activation
    note left of zj : Activation
    note left of END : Output
    classDef hidden display: none;
{{</mermaid>}}


### Model

Linear Regression assumes a **linear relationship** between the target {{< katex >}}y \in \mathbb{R}{{< /katex >}} and the features {{< katex >}}\mathbf{x}\in \mathbb{R}^d{{< /katex >}}.
{{< katex display=true >}}
y = f(\mathbf{x}) = w_1 x_1 + w_2 x_2 + ... + w_d x_d + b = \mathbf{w}^T\mathbf{x} + b,
{{< /katex >}}
where {{< katex >}}\mathbf{w}\in \mathbb{R}^d{{< /katex >}} is the {{< katex >}}d{{< /katex >}}-dimensional *weight vector* and {{< katex >}}b \in \mathbb{R}{{< /katex >}} is the *bias term*.
{{% hint warning %}}
Without loss of generalization we can ignore the bias term as it can be subsumed into the design matrix by appending a column of ones.
{{% /hint %}}
{{< katex display=true >}}
\hat{y} = f(\mathbf{x}) = \mathbf{w}^T\mathbf{x}
{{< /katex >}}
We often stack all the {{< katex >}}n{{< /katex >}} examples into a *design matrix* {{< katex >}}\mathbf{X} \in \mathbb{R^{n \times d}}{{< /katex >}}, where each row is one instance. The predictions for all the {{< katex >}}n{{< /katex >}} instances {{< katex >}}\mathbf{\hat{y}} \in \mathbb{R}^n{{< /katex >}} can be written conveniently as a matrix-vector product.
{{< katex display=true >}}
\mathbf{\hat{y}} = \mathbf{X}\mathbf{w}
{{< /katex >}}

### Statistical model

The probability of {{< katex >}}y{{< /katex >}} for a given feature vector ({{< katex >}}\mathbf{x}\in \mathbb{R}^d{{< /katex >}}) is modelled as
{{< katex display=true >}}
\text{Pr}[y|\mathbf{x},\mathbf{w}] = \mathcal{N}(y|\mathbf{w}^T\mathbf{x},\sigma^2)
{{< /katex >}}
where {{< katex >}}\mathbf{w}\in \mathbb{R}^d{{< /katex >}} are the weights/**parameters** of the model and {{< katex >}}\mathcal{N}{{< /katex >}} is the **normal** distribution with mean {{< katex >}}\mathbf{w}^T\mathbf{x}{{< /katex >}} and variance {{< katex >}}\sigma^2{{< /katex >}}. The model prediction is given by
{{< katex display=true >}}
\text{E}[y|\mathbf{x},\mathbf{w}] = \mathbf{w}^T\mathbf{x}
{{< /katex >}}

### Likelihood

Given a dataset {{< katex >}}\mathcal{D}=\{\mathbf{x}_i \in \mathbb{R}^d,\mathbf{y}_i \in \mathbb{R}\}_{i=1}^N{{< /katex >}} containing {{< katex >}}n{{< /katex >}} examples we need to estimate the parameter vector {{< katex >}}\mathbf{w}{{< /katex >}} by maximizing the likelihood of data.

> In practice we minimize the **negative log likelihood**.

Let {{< katex >}} \hat{y}_i = \mathbf{w}^T\mathbf{x}_i{{< /katex >}} be the model prediction for each example in the training dataset. The negative log likelihood (NLL) is given by
{{< katex display=true >}}
\begin{align}
L(\mathbf{w}) &= - \sum_{i=1}^{N} \log \left[\text{Pr}[y_i|\mathbf{x}_i,\mathbf{w}]\right] \nonumber \\
                       &= \frac{N}{2} \log(2\pi\sigma^2) + \frac{1}{2\sigma^2} \sum_{i=1}^{N} (y_i-\hat{y}_i)^2 \nonumber \\

\end{align}
{{< /katex >}}
This is equivalent to minimizing the **Mean Squared Error** (MSE) loss.
{{< katex display=true >}}
\begin{align}
L(\mathbf{w}) &= \frac{1}{2N} \sum_{i=1}^{N} (y_i-\hat{y}_i)^2 \nonumber \\
\end{align}
{{< /katex >}}
We need to choose the model parameters that optimizes (minimizes) the loss function.
{{< katex display=true >}}
\hat{\mathbf{w}} = \argmin_{\mathbf{w}} L(\mathbf{w})
{{< /katex >}}


### Loss function

**Mean Squared Error** (MSE)
{{< katex display=true >}}
L(\mathbf{w}) = \frac{1}{2N} \sum_{i=1}^{N}  \left(y_i - \mathbf{w}^T\mathbf{x}_i  \right)^2
{{< /katex >}}

{{<button href="https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html#torch.nn.MSELoss">}}torch.nn.MSELoss{{</button>}}


### Gradient
The loss function using matrix notation can be written as follows.
{{< katex display=true >}}
L(\mathbf{w}) = \frac{1}{2} \| \mathbf{y} - \mathbf{X}\mathbf{w} \|^2
{{< /katex >}}
The gradient of the loss function if given by
{{< katex display=true >}}
\nabla_{\mathbf{w}} L(\mathbf{w}) = \mathbf{X}^T(\mathbf{X}\mathbf{w}-\mathbf{y})
{{< /katex >}}


### Analytic solution

Taking the derivative of the loss with respect to {{< katex >}}\mathbf{w}{{< /katex >}} and setting it to zero yields:
{{< katex display=true >}}
\hat{\mathbf{w}} = \left(\mathbf{X}^T\mathbf{X}\right)^{-1}\mathbf{X}^T\mathbf{y}.
{{< /katex >}}
This is unique when the design matrix is full rank, that columns of the design matrix are linearly independent or no features is linearly dependent on the others.

{{% hint warning %}}
In practice we will use mini-batch stochastic gradient descent.
{{% /hint %}}
