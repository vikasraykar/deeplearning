---
title: Gradient Descent
weight: 3
bookToc: true
---

# Gradient Descent

> Steepest descent.

## Batch Gradient Descent

We take a small step in the direction of the **negative gradient**.

{{< katex display=true >}}
\mathbf{w}^t \leftarrow \mathbf{w}^{t-1} - \eta \nabla L(\mathbf{w}^{t-1})
{{< /katex >}}

The parameter {{< katex >}}\eta > 0{{< /katex >}} is called the **learning rate**.

Each step requires that the **entire training data** be processed to compute the gradient {{< katex >}}\nabla L(\mathbf{w}^{t-1}){{< /katex >}}. For large datasets this is not comptationally efficient.

This update is repeated multiple times (till covergence).

```python
for epoch in range(n_epochs):
  dw = gradient(loss, data, w)
  w = w - learning_rate * dw
```

## Stochastic Gradient Descent

In general most loss functions can be written as sum over each training instance.
{{< katex display=true >}}
L(\mathbf{w}) = \sum_{i=1}^{N} L_i(\mathbf{w})
{{< /katex >}}

In Stochastic Gradient Descent (SGD) we update the parameters **one data point at a time**.
{{< katex display=true >}}
\mathbf{w}^t \leftarrow \mathbf{w}^{t-1} - \eta \nabla L_i(\mathbf{w}^{t-1})
{{< /katex >}}

> A complete passthrough of the whole dataset is called an **epoch**. Training is done for multiple epochs depending on the size of the dataset.

```python
for epoch in range(n_epochs):
  for example in data:
    dw = gradient(loss, example, w)
    w = w - learning_rate * dw
```


{{% hint warning %}}
Bottou, L. (2010). [Large-Scale Machine Learning with Stochastic Gradient Descent](https://leon.bottou.org/publications/pdf/compstat-2010.pdf). In: Lechevallier, Y., Saporta, G. (eds) Proceedings of COMPSTAT'2010. Physica-Verlag HD.
{{% /hint %}}


## Mini-batch Stochastic Gradient Descent

Using a single examples results in a very noisy estimate of the gradient. So we use a small subset of data called **mini-batch** of size B(**batch size**) to compute the gradient.

{{< katex display=true >}}
\mathbf{w}^t \leftarrow \mathbf{w}^{t-1} - \eta \nabla L_{batch}(\mathbf{w}^{t-1})
{{< /katex >}}


```python
for epoch in range(n_epochs):
  for mini_batch in get_batches(data, batch_size):
    dw = gradient(loss, mini_batch, w)
    w = w - learning_rate * dw
```

{{% hint info %}}
Mini-batch SGD is the most commonly used method and is sometimes refered to as just SGD.
- Typical choices of the batch size are B=32,64,128,256,..
- In practice we do a random shuffle of the data per epoch.
{{% /hint %}}

## Learning rate schedule

## Learning curve

## Training loop

## Momentum

## Adaptive Learning Rates

## Adagrad

## RMSProp

## Adam



