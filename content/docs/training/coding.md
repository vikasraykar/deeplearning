---
title: Coding
weight: 8
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

> Linear Regression with numpy and batch gradient descent.

In the first coding assigment you will be implementing a basic **Linear Regression** model from scratch using **only `numpy`**. You will be implementing a basic batch gradient descent optimizer.

{{% hint danger %}}
You can use only `numpy` and are not allowed to to use `torch` or any other python libraries.
{{% /hint %}}


1. Review the [linear regression model](/deeplearning/docs/training/model/#linear-regression) and its loss function.
1. Given a feature matrix {{< katex >}}\mathbf{X}{{< /katex >}} as a {{< katex >}}N \times d{{< /katex >}} `numpy.ndarray` write the prediction and the loss function using matrix notation and carefully check for dimensions.
1. Accont for the bias by appending the feature matrix with a column of ones.
1. Derive the expression for the gradient of the loss function.
1. Implement a basic [batch gradient descent optimizer](/deeplearning/docs/training/gradient_descent/#batch-gradient-descent) with a fixed learning rate first.
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

    def predict(self, X: np.ndarray) -> np.ndarray:
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

### Problem 2

> Logistic Regression with numpy and batch gradient descent.

In the second coding assigment you will be implementing a basic **Logisitc Regression** model from scratch using **only `numpy`**. You will be implementing a basic batch gradient descent optimizer.

{{% hint danger %}}
You can use only `numpy` and are not allowed to to use `torch` or any other python libraries.
{{% /hint %}}


1. Review the [logistic regression model](/deeplearning/docs/training/model/#llogistic-regression) and its loss function.
1. Given a feature matrix {{< katex >}}\mathbf{X}{{< /katex >}} as a {{< katex >}}N \times d{{< /katex >}} `numpy.ndarray` write the prediction and the loss function using matrix notation and carefully check for dimensions.
1. Accont for the bias by appending the feature matrix with a column of ones.
1. Derive the expression for the gradient of the loss function. Modularize it so that it should look exactly like the derivative you got for linear regression.
1. Implement a basic [batch gradient descent optimizer](/deeplearning/docs/training/gradient_descent/#batch-gradient-descent) with a fixed learning rate first.
1. Track the loss every few epochs and check the actual and estimated parameters.
1. Check the accuracy on the train and the test set.
1. Implement a simple learning rate decay as follows `lr=lr/(1+decay_factor*epoch)`.

> A sample stub is provided in the repo as below. Your task is to implement the `predict_proba`, `predict` and the `fit` function.

{{<button href="https://github.com/vikasraykar/deeplearning-dojo/blob/main/stubs/LogisticRegressionNumpy.py">}}LogisticRegressionNumpy.py{{</button>}}

```python
"""Basic implementation of Logistic Regression using numpy only."""

import numpy as np

class LogisticRegression:
    """Logistic Regression"""

    def __init__(self):
        pass

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        learning_rate: float = 1e-3,
        learning_rate_decay: bool = True,
        learning_rate_decay_factor: float = 1.0,
        num_epochs: int = 100,
        track_loss_num_epochs: int = 10,
    ):
        """Train.

        Args:
            X_train (np.ndarray): Feature matrix. (N,d)
            y_train (np.ndarray): Target labels (0,1). (N,)
            learning_rate (float, optional): The initial learning rate. Defaults to 1e-3.
            learning_rate_decay (bool, optional): If True enables learning rate decay. Defaults to True.
            learning_rate_decay_factor (float, optional): The learning rate decay factor (1/(1+decay_factor*epoch)). Defaults to 1.0.
            num_epochs (int, optional): The number of epochs to train. Defaults to 100.
            track_loss_num_epochs (int, optional): Compute loss on training set once in k epochs. Defaults to 10.
        """
        pass

    def predict_proba(self, X: np.ndarray):
        """Predict the probability of the positive class (Pr(y=1)).

        Args:
            X (np.ndarray): Feature matrix. (N,d)

        Returns:
            y_pred_proba (np.ndarray): Predicted probabilities. (N,)
        """
        pass

    def predict(
        self,
        X: np.ndarray,
        threshold: float = 0.5,
    ):
        """Predict the label(0,1).

        Args:
            X (np.ndarray): Feature matrix. (N,d)
            add_colum_vector_for_bias (bool, optional): Add a column vector of ones to model the bias term. Defaults to True.
            threshold (float, optional): The threshold on the probabilit. Defaults to 0.5.

        Returns:
            y (np.ndarray): Prediction vector. (N,)
        """
        pass
```

### Problem 3

> Logistic Regression with torch and min-batch SGD.

- Review [Datasets & DataLoaders](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html) in pytorch.
- Experiment withe different optimizers covered in this lectures and plot the learning curve for different optimizers (SGD, SGD with momentum, AdaGrad, RMSProp, Adam), AdamW.
- Tune the learning rate for a optimizer.

> A sample stub is provided in the repo as below.

{{<button href="https://github.com/vikasraykar/deeplearning-dojo/blob/main/stubs/LogisticRegressionPytorch.py">}}LogisticRegressionPytorch.py{{</button>}}

### Problem 4

> Logistic Regression with torch and min-batch SGD on a publicly avaiable dataset.

Chosee one publicly avaiable large dataset and implement custom datsets and loaders and learn either a linear regression model.

[Real Estate Data UAE](https://www.kaggle.com/datasets/kanchana1990/real-estate-data-uae)

### Bonus problem

> AdamW implementation

Implement the AdamW optimizer as a subclass of `torch.optim.Optimizer`.

An Optimizer subclass much implemnt two methods.

```python
import torch

class AdamW(torch.optim.Optimizer):
    def __init__(self,params, ...):
        pass

    def step(self):
        pass
```

{{% hint danger %}}
The PyTorch optimizer API has a few subtleties and study how some optimizers are written.
{{% /hint %}}
