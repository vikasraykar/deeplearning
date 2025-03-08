---
title: Coding
weight: 9
bookToc: true
---

## Coding assignment

{{< katex >}}{{< /katex >}}

### Setup

https://github.com/vikasraykar/deeplearning-dojo/

```
git clone https://github.com/vikasraykar/deeplearning-dojo.git
cd deeplearning-dojo

python -m venv .env
source .env/bin/activate
pip install -r requirements.txt
```

### Problem 1

> Linear Regression with numpy.

In the first coding assigment you will be implementing a basic **Linear Regression** model from scratch using **only `numpy`**. You will be implementing a basic batch gradient descent optimizer.

{{% hint danger %}}
You can use only `numpy` and are not allowed to to use `torch` or any other python libraries.
{{% /hint %}}


1. Review the [linear regression model](/docs/training/model/#linear-regression) and its loss function.
1. Given a feature matrix {{< katex >}}\mathbf{X}{{< /katex >}} as a {{< katex >}}N \times d{{< /katex >}} `numpy.ndarray` write the prediction and the loss function using matrix notation and carefully check for dimensions.
1. Accont for the bias by appending the feature matrix with a column of ones.
1. Derive the expression for the gradient of the loss function.
1. Implement a basic [batch gradient descent optimizer](training/gradient_descent/#batch-gradient-descent) with a fixed learning rate first.
1. Track the loss every few epochs and check the actual and estimated parameters.
1. Check the MSE loss on the train and the test set.
1. Implement a simple learning rate decay as follows `lr=lr/(1+decay_factor*epoch)`.

> A sample stub is provided in the repo as below. Your task is to implement the `predict` and the `fit` function.

{{<button href="https://github.com/vikasraykar/deeplearning-dojo/blob/main/stubs/LinearRegressionNumpy.py">}}LinearRegressionNumpy.py{{</button>}}

```python
"""Basic implementation of Linear Regression using only numpy.
"""
import numpy as np

class LinearRegression:
    """Linear Regression."""

    def __init__(self):
        pass

    def pred(self, X: np.ndarray) -> np.ndarray:
        """Predction.

        Args:
            X (np.ndarray): Features matrix. (N,d)
            add_colum_vector_for_bias (bool, optional): Add a column vector of ones to model
            the bias term. Defaults to True.

        Returns:
            y (np.ndarray): Prediction vector. (N,)
        """
        pass

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        learning_rate: float = 1e-3,
        learning_rate_decay: bool = True,
        learning_rate_decay_factor: float = 1.0,
        num_epochs: int = 100,
        track_loss_num_epochs: int = 100,
    ):
        """Training.

        Args:
            X_train (np.ndarray): Features matrix. (N,d)
            y_train (np.ndarray): Target vector. (N,)
            learning_rate (float, optional): Learning rate. Defaults to 0.001.
            learning_rate_decay (bool, optional): If True does learning rate deacy. Defaults to True.
            learning_rate_decay_factor (float, optional): The deacay factor (lr=lr/(1+decay_factor*epoch)). Defaults to 1.0.
            num_epochs (int, optional): Number of epochs. Defaults to 100.
            track_loss_num_epochs (int, optional): Compute loss on training set once in k epochs. Defaults to 100.
        """
        pass
```

## numpy

Derive the equation for the gradient of the BCE loss used in logistic regression.

## pytorch

Plot learning curve for 3 differnt optimizers.
## Adam implementation
