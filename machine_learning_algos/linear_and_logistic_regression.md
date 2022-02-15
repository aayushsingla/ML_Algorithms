<!-- TODO  -->

# Linear and Logistic Regression

Part of: <https://github.com/aayushsingla/ML_Algorithms>

<!-- TOC -->

- [Linear regression](#linear-regression)
  - [How to determine coefficients?](#how-to-determine-coefficients)
  - [Implementation of Linear Regression using Gradient Descent](#implementation-of-linear-regression-using-gradient-descent)
  - [Advantages and Disadvantages](#advantages-and-disadvantages)
- [Logistic regression](#logistic-regression)
  - [Types of Logistic regression](#types-of-logistic-regression)
  - [Implementation of Logistic Regression using Gradient Descent](#implementation-of-logistic-regression-using-gradient-descent)
  - [Implementation of Multi-logistic Regression using Gradient Descent](#implementation-of-multi-logistic-regression-using-gradient-descent)
- [What can we improve here?](#what-can-we-improve-here)
- [References](#references)

<!-- /TOC -->

## Linear regression

Building an ML model is basically mapping input X to target Y using a **hypothesis function**. When the target variable Y is continuous i.e. it can take any value and there is no finite bound on number of values it can take, the problem that you are solving is **regression problem**. When your target variable is discrete and you require your example to take only one of the available discrete values (and so there is a finite bound on the number of values it can take), the problem you are solving is **classification problem** because you are classifying your example into one or more of those available discrete values/classes.

Linear Regression says that conditional expectation of **Y** given **X** i.e. **E(Y|X)** can be expressed as a linear combination of features. Regression can be linear or non-linear. What linear regression means is that it’s linear in parameters. So you could have features like **X**, **X^2**, **X^3** in your data but the coefficients **β** will be linear.  

`E(Y|X) = β1.X + β2.X^2 + β3.X^3`

#### How to determine coefficients?

There is a closed-form solution that exists for getting regression coefficients and is obtained with minimizing Ordinary Least Squares (OLS). This is given by:  

`β = (1/(X_transpose * X)) * X_transpose * Y`

This is called computing using normal-equations and the complexity is given by O(d^3), while **complexity using Gradient Descent is given by _O(d^2.n)_**, where _d_ is the number of features and _n_ is the number of examples.

#### Implementation of Linear Regression using Gradient Descent

```python
class LinearRegression:
    def __init__(self, w_init=0, b_init=0):
        # hyper-parameters
        self.alpha, self.beta = 1.e-3, 1.e-3 # learning rates
        self.w_init = w_init   # iniital value of weight w
        self.b_init = b_init   # iniital value of bias b
        self.epsilon = 1.e5    # accuracy to which coefficients/params
                               # need to be tuned.

    def loss(self, Y_pred, Y):
        # MSE loss - mean squared loss
        return sum((Y_pred - Y)**2)/len(Y)

    def gradient(self, Y_pred, Y, X):
        # change in loss w.r.t to w and b
        # ie. partial derivative of loss w.r.t to w and b
        # loss we have is MSE = (y - (w.x+b))^2
        # d(loss)/dw = 2*(y-(w.x+b))*(-x)
        # d(loss)/db = 2*(y-(w.x+b))*(-1)
        dw = 2 * (Y-Y_pred) * (-X)
        db = 2 * (Y-Y_pred) * (-1)
        return dw, db

    def linear_regression(self, X, Y):
        # main function that performs linear regression
        # Implementation of y = w.x + b
        w, b = self.w_init, self.b_init
        epsilon = self.epsilon

        # book keeping arrays (to track the history)
        W, B, costs = [], [], []

        while abs(dw)>epsilon and abs(db)>epsilon:
            # obtain predictions using current vars.
            Y_pred = w*X + b
            # difference bewtween predictions and true values.
            loss = self.loss(Y_pred, Y)
            # gradient ie. change in loss based on hyper-parameters
            dw, db = self.gradient(Y_pred, Y, X)
            # update the variables
            w = w - (self.alpha * dw)
            b = b - (self.beta * db)
            # updade the history
            costs.append(loss)
            W.append(w)
            B.append(b)

        return W, B, costs
```

#### Advantages and Disadvantages

Linear Regression comes with a set of assumptions and often they don’t hold much with practical data. This requires validation of assumptions and also may require some data pre-processing, which is why they are not used much (in competitions). However, LR models are highly explainable because, at the end of the day, you come up with an equation like above. So it’s useful in critical business areas (like Finance) where you can’t afford to go wrong and trust your model.


## Logistic regression
Let's start with why can't we use Linear regression directly for categorical data (like just predict a continous bounded number between 0 and number_of_labels):
1. Regression would assume the target variable as numeric which is categorical.
2. Since it treats the variable as numeric, there is no limit on number of values it can take, so it can predict any number after learning the hypothesis but this isn’t correct since we require our model to predict from the given set of nominal/ordinal values.
3. Can’t be applied to textual data (for example - Sentiment Analysis) since Text is non-numeric.

#### Types of Logistic regression
#### Implementation of Logistic Regression using Gradient Descent
#### Implementation of Multi-logistic Regression using Gradient Descent


## What can we improve here?
<!-- [Improve] -->
- **Notebook:** [Linear Regression using SGD](https://github.com/rajatguptakgp/ml_from_scratch/blob/master/linear_regression_sgd.ipynb)

- **Notebook:** [LR with Least Absolute Deviation](https://github.com/rajatguptakgp/ml_from_scratch/blob/master/lad_linear_regression.ipynb)

- **Notebook:** [Multiple Linear Regression](https://github.com/rajatguptakgp/ml_from_scratch/blob/master/multiple_linear_regression.ipynb)
- **Notebook:** [Implement Multiple Logistic Regression from scratch](https://github.com/rajatguptakgp/ml_from_scratch/blob/master/multiple_logistic_regression.ipynb)

## References
1. [Rajat Gupta ML Notes](https://ml-notes-rajatgupta.notion.site/)
