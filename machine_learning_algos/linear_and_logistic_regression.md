# Linear and Logistic Regression

Part of: <https://github.com/aayushsingla/ML_Algorithms>

<!-- TOC -->

-   [Linear and Logistic Regression](#linear-and-logistic-regression)
    -   [Linear regression](#linear-regression)
        -   [How to determine coefficients?](#how-to-determine-coefficients)
        -   [Implementation of Linear Regression using Gradient Descent](#implementation-of-linear-regression-using-gradient-descent)
        -   [Advantages and Disadvantages](#advantages-and-disadvantages)
    -   [Logistic regression](#logistic-regression)
        -   [Generalized Linear Models (GLM)](#generalized-linear-models-glm)
        -   [Implementation of Logistic Regression using Gradient Descent](#implementation-of-logistic-regression-using-gradient-descent)
        -   [Types of classification](#types-of-classification)
        -   [Multi-class classification using logistic regression](#multi-class-classification-using-logistic-regression)
        -   [Multi-label classification using logisitic regression](#multi-label-classification-using-logisitic-regression)
    -   [What can we improve here?](#what-can-we-improve-here)
    -   [References](#references)

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
            # update the history
            costs.append(loss)
            W.append(w)
            B.append(b)

        return W, B, costs
```

#### Advantages and Disadvantages

Linear Regression comes with a set of assumptions and often they don’t hold much with practical data. This requires validation of assumptions and also may require some data pre-processing, which is why they are not used much (in competitions). However, LR models are highly explainable because, at the end of the day, you come up with an equation like above. So it’s useful in critical business areas (like Finance) where you can’t afford to go wrong and trust your model.

## Logistic regression

Let's start with why can't we use Linear regression directly for categorical data (like just predict a continuous bounded number between 0 and number_of_labels):
1. Regression would assume the target variable as numeric which is categorical.
2. Since it treats the variable as numeric, there is no limit on number of values it can take, so it can predict any number after learning the hypothesis but this isn’t correct since we require our model to predict from the given set of nominal/ordinal values.
3. Can’t be applied to textual data (for example - Sentiment Analysis) since Text is non-numeric.

In equation, `p = f(β1.X + β2.X^2 + β3.X^3)` the function argument on right side can take real values but the left side which is probability score take values in [0, 1], so we need a function such that this gets satisfied. So we use the logit of **p** which takes real values to link both sides of equation. The logit link function is used to model the probability of 'success' as a function of covariates. Mathematically, the logit is the inverse of the [standard logistic function](https://en.wikipedia.org/wiki/Logistic_function).

`log(p/1-p) = (β1.X + β2.X^2 + β3.X^3)`

p stands for probability of occurrence of say class 1 (out of two classes 1/0 or one-vs-rest in multi-class). In a binary classification, if you do a label switch, the change in equation is that the coefficients change signs, regression coefficient magnitudes remain same.

#### Generalized Linear Models (GLM)

**Generalized Linear Models (GLM)** is generalized form of linear regression which can help linking LHS and RHS from different domains using **Link Function**. Essentially, linking conditional expectation **E(Y|X)** with linear combination of features **βX** using the link function **g**:  

`E(Y|X) = μ = g^-1(βX)`

In logistic regression, the link function is logit link like we discussed above.

Logistic Regression is a parametric model similar to Linear Regression as we can express the hypothesis function using a finite set of parameters. However, unlike Linear Regression, the **Least-Squares loss is not convex for classification** and so we use **Negative Log-Likelihood (NLL)** or **Cross-Entropy loss**. A non-convex loss function doesn't guarantee loss minimization and hence, is avoided.

#### Implementation of Logistic Regression using Gradient Descent

```py
class LogisticRegression:
    def __init__(self, w_init=0, b_init=0):
        # hyper-parameters
        self.alpha, self.beta = 1.e-3, 1.e-3 # learning rates
        self.w_init = w_init   # iniital value of weight w
        self.b_init = b_init   # iniital value of bias b
        self.epsilon = 1.e5    # accuracy to which coefficients/params
                               # need to be tuned.

    def loss(self, Y_pred, Y_true):
        # Binary cross entropy - log loss
        # Args:
        # Y_true: Actual probability of success
        # Y_pred: predicted probability of success
        loss = -1 * ((Y_true * np.log(Y_pred) + ((1-Y_true) * np.log(1-Y_pred))))
        return sum(loss)/len(Y)

    def gradient(self, Y_pred, Y, X):
        # change in loss w.r.t to w and b
        # ie. partial derivative of loss w.r.t to w and b
        # loss we have is -(Y.log(logistic) + (1-Y).log(1-(logistic))
        # d(loss)/dw = ((1/1+e^(-Y_pred))-Y)*X [pick your N.B. (DIY)]
        # d(loss)/db = 1/1+e^(-Y_pred)
        dw = X * ((1 / 1 + np.exp(-Y_pred))-Y)
        db = (1/(1 + np.exp(-Y_pred))) - Y
        return dw, db

    def logistic_regression(self, X, Y):
        # main function that performs linear regression
        # Implementation of y = w.x + b
        w, b = self.w_init, self.b_init
        epsilon = self.epsilon

        # book keeping arrays (to track the history)
        W, B, costs = [], [], []

        while abs(dw)>epsilon and abs(db)>epsilon:
            # obtain predictions using current vars.
            # log(p/1-p) = wX + b
            # 1/p - 1 = e^-(wX+b)
            # p = 1/(1+e^(-wX+b))
            Y_pred = 1 + (1 + np.exp(-(w*X+b)))

            # difference between predictions and true values.
            loss = self.loss(Y_pred, Y)
            # gradient ie. change in loss based on hyper-parameters
            dw, db = self.gradient(Y_pred, Y, X)
            # update the variables
            w = w - (self.alpha * dw)
            b = b - (self.beta * db)
            # update the history
            costs.append(loss)
            W.append(w)
            B.append(b)

        return W, B, costs
```

#### Types of classification

Classification can be binary, multi-class or multi-label:

-   **Binary Classification**: Simplest form of classification where the output is binary. Whether a person defaulted or not, a product is returned or not.
-   **Multi-class classification**: This means that every example can take a single class and there exist a number of classes that can be assigned to the example. For example - an object detected can be either person, car, bus or any other class but it cannot be person and car both.
-   **Multi-label classification**: This means that a single example can have multiple labels of different classes in it. For example - presence of a dog, cat and bird in an image.

#### Multi-class classification using logistic regression

Logistic regression by default is designed for binary classification. Some extensions like one-vs-rest can allow logistic regression to be used for multi-class classification problems, although they require that the classification problem first be transformed into multiple binary classification problems. An alternate approach involves changing the logistic regression model to support the prediction of multiple class labels directly. Specifically, to predict the probability that an input example belongs to each known class label.

The probability distribution that defines multi-class probabilities is called a multinomial probability distribution. A logistic regression model that is adapted to learn and predict a multinomial probability distribution is referred to as **Multinomial Logistic Regression**. Similarly, we might refer to default or standard logistic regression as Binomial Logistic Regression.

-   Binomial Logistic Regression: Standard logistic regression that predicts a binomial probability (i.e. for two classes) for each input example.
-   Multinomial Logistic Regression: Modified version of logistic regression that predicts a multinomial probability (i.e. more than two classes) for each input example.

**Changing logistic regression from binomial to multinomial probability requires a change to the loss function used to train the model (e.g. log loss to cross-entropy loss), and a change to the output from a single probability value to one probability for each class label.**

#### Multi-label classification using logisitic regression

Like with multi-class classification, logistic regression can be modelled to solve mutli-label classification. This method can be carried out in three different ways as:
1. **Binary Relevance**: This is the simplest technique, which basically treats each label as a separate single class classification problem, like what we have studied till now. It is most simple and efficient method but the only drawback of this method is that it doesn’t consider labels correlation because it treats every target variable independently.
2. **Classifier Chains**: Like the name suggests, in this method we have a **N** classifiers in series (like a chain) for **N** labels. The input to the first classifier is the features and it handles the problem as single-class classification for label 1. Classifier 2 in series takes in output of classifier 1 along with features and handles the problem as single-class classification for label 2. This goes on until the result for all labels are known by the end of the chain.
3. **Label powerset**: This is the most interesting approach out of all three. In this problem transformation, we convert multi-label classification to multi-class classification problem. Let's say we have 3 labels (X, Y, Z). So to solve this using multi-class classification, we consider all the possibilities involving these labels as separate classes. For ex: we will have 2^3 classes ie. ["-", "X", "Y", "Z", "XY", "YZ", "ZX", "XYZ"]. So we expect that input whose labels are X and Y should belong to class "XY" and input whose labels are X, Y and Z should belong to "XYZ" class. The issue with this approach is that as the training data increases, number of classes become more. Thus, increasing the model complexity, and would result in a lower accuracy.

**Note**: KNN and Ensemble algorithms are also used for multi-label classiification.  
**Also**: Multiple Linear Regression and Multiple Logistic Regression refers to a probems whose input features have multiple variables rather than a single independent variable. These are implemented using matrix operations (mostly using numpy) and not floating point operations.

## What can we improve here?

<!-- [Improve] -->

-   **Notebook:** [Linear Regression using SGD](https://github.com/rajatguptakgp/ml_from_scratch/blob/master/linear_regression_sgd.ipynb)

-   **Notebook:** [LR with Least Absolute Deviation](https://github.com/rajatguptakgp/ml_from_scratch/blob/master/lad_linear_regression.ipynb)

-   **Notebook:** [Multiple Linear Regression](https://github.com/rajatguptakgp/ml_from_scratch/blob/master/multiple_linear_regression.ipynb)
-   **Notebook:** [Implement Multiple Logistic Regression from scratch](https://github.com/rajatguptakgp/ml_from_scratch/blob/master/multiple_logistic_regression.ipynb)

## References

1.  [Rajat Gupta ML Notes](https://ml-notes-rajatgupta.notion.site/)
2.  [Machine learning mastery](https://machinelearningmastery.com/multinomial-logistic-regression-with-python/)
3.  [Analytics vidhya](https://www.analyticsvidhya.com/blog/2017/08/introduction-to-multi-label-classification/)
