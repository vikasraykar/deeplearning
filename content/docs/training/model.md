---
title: Models
weight: 2
bookToc: true
---

## Single Layer Networks

> For simplicity for this chapter we will mainly introduce single layer networks for regression and classification.

### Linear Regression
Linear Regression is a single layer neural network for regression. The probability of {{< katex >}}y{{< /katex >}} for a given feature vector ({{< katex >}}\mathbf{x}\in \mathbb{R}^d{{< /katex >}}) is modelled as
{{< katex display=true >}}
\text{Pr}[y|\mathbf{x},\mathbf{w}] = \mathcal{N}(y|\mathbf{w}^T\mathbf{x},\sigma^2)
{{< /katex >}}
where {{< katex >}}\mathbf{w}\in \mathbb{R}^d{{< /katex >}} are the weights/**parameters** of the model and {{< katex >}}\mathcal{N}{{< /katex >}} is the **normal** distribution with mean {{< katex >}}\mathbf{w}^T\mathbf{x}{{< /katex >}} and variance {{< katex >}}\sigma^2{{< /katex >}}. The prediction is given by
{{< katex display=true >}}
\text{E}[y|\mathbf{x},\mathbf{w}] = \mathbf{w}^T\mathbf{x}
{{< /katex >}}
{{% hint warning %}}
Without loss of generalization we ignore the bias term as it can be incorporated into the feature vector.
{{% /hint %}}
Given a dataset {{< katex >}}\mathcal{D}=\{\mathbf{x}_i \in \mathbb{R}^d,\mathbf{y}_i \in \mathbb{R}\}_{i=1}^N{{< /katex >}} containing {{< katex >}}n{{< /katex >}} examples we need to estimate the parameter vector {{< katex >}}\mathbf{w}{{< /katex >}} by maximizing the likelihood of data.

> In practice we minimize the **negative log likelihood**.

Let {{< katex >}} \mu_i = \mathbf{w}^T\mathbf{x}_i{{< /katex >}} be the model prediction for each example in the training dataset. The negative log likelihood (NLL) is given by
{{< katex display=true >}}
\begin{align}
L(\mathbf{w}) &= - \sum_{i=1}^{N} \log \left[\text{Pr}[y_i|\mathbf{x}_i,\mathbf{w}]\right] \nonumber \\
                       &= \frac{N}{2} \log(2\pi\sigma^2) + \frac{1}{2\sigma^2} \sum_{i=1}^{N} (y_i-\mu_i)^2 \nonumber \\

\end{align}
{{< /katex >}}
This is equivalent to minimizing the **Mean Squared Error** (MSE) loss.
{{< katex display=true >}}
\begin{align}
L(\mathbf{w}) &= \frac{1}{N} \sum_{i=1}^{N} (y_i-\mu_i)^2 \nonumber \\
\end{align}
{{< /katex >}}
We need to choose the model parameters that optimizes (minimizes) the loss function.
{{< katex display=true >}}
\hat{\mathbf{w}} = \argmin_{\mathbf{w}} L(\mathbf{w})
{{< /katex >}}

{{<button href="https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html#torch.nn.MSELoss">}}torch.nn.MSELoss{{</button>}}

### Logistic Regression
Logisitc Regression is a single layer neural network for binary classification. The probability of the positive class ({{< katex >}}y=1{{< /katex >}}) for a given feature vector ({{< katex >}}\mathbf{x}\in \mathbb{R}^d{{< /katex >}}) is given by
{{< katex display=true >}}
\text{Pr}[y=1|\mathbf{x},\mathbf{w}] = \sigma(\mathbf{w}^T\mathbf{x})
{{< /katex >}}
where {{< katex >}}\mathbf{w}\in \mathbb{R}^d{{< /katex >}} are the weights/**parameters** of the model and {{< katex >}}\sigma{{< /katex >}} is the **sigmoid** activation function defined as
{{< katex display=true >}}
\sigma(x) = \frac{1}{1-e^{-z}}
{{< /katex >}}
{{% hint warning %}}
Without loss of generalization we ignore the bias term as it can be incorporated into the feature vector.
{{% /hint %}}
Given a dataset {{< katex >}}\mathcal{D}=\{\mathbf{x}_i \in \mathbb{R}^d,\mathbf{y}_i \in [0,1]\}_{i=1}^N{{< /katex >}} containing {{< katex >}}n{{< /katex >}} examples we need to estimate the parameter vector {{< katex >}}\mathbf{w}{{< /katex >}} by maximizing the likelihood of data.

> In practice we minimize the **negative log likelihood**.

Let {{< katex >}} \mu_i = \text{Pr}[y_i=1|\mathbf{x}_i,\mathbf{w}] = \sigma(\mathbf{w}^T\mathbf{x}_i){{< /katex >}} be the model prediction for each example in the training dataset. The the negative log likelihood (NLL) is given by
{{< katex display=true >}}
\begin{align}
L(\mathbf{w}) &= - \sum_{i=1}^{N} \log\left[\mu_i^{y_i}(1-\mu_i)^{1-y_i}\right] \nonumber \\
                       &= - \sum_{i=1}^{N} \left[ y_i\log(\mu_i) + (1-y_i)\log(1-\mu_i) \right] \nonumber \\

\end{align}
{{< /katex >}}
This is referred to as the **Binary Cross Entropy** (BCE) loss. We need to choose the model parameters that optimizes (minimizes) the loss function.
{{< katex display=true >}}
\hat{\mathbf{w}} = \argmin_{\mathbf{w}} L(\mathbf{w})
{{< /katex >}}

{{<button href="https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html#torch.nn.BCELoss">}}torch.nn.BCELoss{{</button>}} {{<button href="https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html#torch.nn.BCEWithLogitsLoss">}}torch.nn.BCEWithLogitsLoss{{</button>}}



{{% details "Entropy" %}}
{{< katex >}}{{< /katex >}}The entropy of a discrete random variable {{< katex >}}X{{< /katex >}} with {{< katex >}}K{{< /katex >}} states/categories with distribution {{< katex >}}p_k = \text{Pr}(X=k){{< /katex >}} for {{< katex >}}k=1,...,K{{< /katex >}}  is a measure of uncertainty and is defined as follows.
{{< katex display=true >}}H(X) = \sum_{k=1}^{K} p_k \log_2 \frac{1}{p_k} = - \sum_{k=1}^{K} p_k \log_2 p_k {{< /katex >}}
{{< katex >}}{{< /katex >}}The term {{< katex >}}\log_2\frac{1}{p}{{< /katex >}} quantifies the notion or surprise or uncertainty and entropy is the average uncertainty. The unit is bits ({{< katex >}}\in [0,\log_2 K]{{< /katex >}}) (or nats incase of natural log). The discrete distribution with maximum entropy ({{< katex >}}\log_2 K{{< /katex >}}) is uniform. The discrete distribution with minimum entropy ({{< katex >}}0{{< /katex >}}) is any delta function which puts all mass on one state/category.

Binary entropy

{{< katex >}}{{< /katex >}}For a binary random variable {{< katex >}}X \in {0,1}{{< /katex >}} with {{< katex >}}\text{Pr}(X=1) = \theta{{< /katex >}} and {{< katex >}}\text{Pr}(X=0) = 1-\theta{{< /katex >}} the entropy is as follows.

{{< katex display=true >}}H(\theta) = - [ \theta \log_2 \theta + (1-\theta) \log_2 (1-\theta) ] {{< /katex >}}

{{< katex display=true >}}H(\theta) \in [0,1]{{< /katex >}} and is maximum when {{< katex >}}\theta=0.5{{< /katex >}}.

Cross entropy

{{< katex >}}{{< /katex >}}Cross entropy is the average number of bits needed to encode the data from from a source {{< katex >}}p{{< /katex >}} when we model it using {{< katex >}}q{{< /katex >}}.

{{< katex display=true >}}H(p,q) = - \sum_{k=1}^{K} p_k \log_2 q_k {{< /katex >}}

{{% /details %}}


## Loss functions

https://pytorch.org/docs/stable/nn.html#loss-functions
