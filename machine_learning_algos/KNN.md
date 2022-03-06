# K-Nearest Neighbours

Part of: <https://github.com/aayushsingla/ML_Algorithms>

<!-- TOC -->

- [K-Nearest Neighbours](#k-nearest-neighbours)
  - [Introduction](#introduction)
  - [Preparation of data for KNN](#preparation-of-data-for-knn)
  - [Bias-Variance tradeoff for K](#bias-variance-tradeoff-for-k)
  - [How to choose value of K?](#how-to-choose-value-of-k)
  - [Possible distance metrics that can be used:](#possible-distance-metrics-that-can-be-used)
  - [Advantages of KNN](#advantages-of-knn)
  - [Downsides of KNN](#downsides-of-knn)
  - [Implementation of KNN from scratch](#implementation-of-knn-from-scratch)
  - [Interview questions](#interview-questions)
  - [What can we improve here?](#what-can-we-improve-here)
  - [References](#references)

<!-- /TOC -->

#### Introduction

**K-Nearest Neighbours (KNN)** is a Supervised ML algorithm that works on the principle that data points that are closer to each other in a coordinate space are similar to each other. It is a versatile algorithm also used for **imputing missing values and resampling datasets**. As the name (K Nearest Neighbor) suggests it considers K Nearest Neighbors (Data points) to predict the class or continuous value for the new Datapoint. It’s a distance-based algorithm that can be used for both regression problems - taking average of values of K nearest neighbours and classification problems - taking majority vote of classes among K nearest neighbours.

Unlike most of the ML methods, KNN is a non-parameteric method and doesnot learn a predefined form of the mapping function. Here we do not learn weights from training data to predict output (as in model-based algorithms) but use entire training instances to predict output for unseen data (for every query, we need whole dataset all the time).

Distance metric and K value are two important considerations for KNN algorithm to work. K is the number of datapoints/neighbours that the algorithms considers before predicting the output. Distance metric decides how the distance between two points will be measured. Euclidean distance is the most popular distance metric. You can also use Hamming distance, Manhattan distance, Minkowski distance as per your need.

For predicting class/ continuous value for a new data point, KNN algorithm considers all the data points in the training dataset. Finds new data point’s ‘K’ Nearest Neighbors (Data points) from feature space and their class labels or continuous values. Then:

-   **For classification**: A class label assigned to the majority of K Nearest Neighbors from the training dataset is considered as a predicted class for the new data point.
-   **For regression**: Mean or median of continuous values assigned to K Nearest Neighbors from training dataset is a predicted continuous value for our new data point

#### Preparation of data for KNN

-   **Data Scaling**: To locate the data point in multidimensional feature space, it would be helpful if all features are on the same scale. As there are more dimensions, two data points need to be closer in every dimension such that the distance between them remains small. If there is even a single dimension that separates them largely, the distance magnitude becomes large and so they are not really close. Hence, we generally normalize or standardize the data before applying KNN to take care of features that have higher variance relative to other features.
-   **Dimensionality Reduction**: KNN suffers from the **curse of dimensionality**. KNN does not work well if there are too many features. As there are more dimensions, two data points need to be closer in every dimension such that the distance between them remains small.  Hence dimensionality reduction techniques like feature selection, principal component analysis can be implemented.
-   **Missing value treatment**: If out of M features one feature data is missing for a particular example in the training set then we cannot locate or calculate distance from that point. Therefore deleting that row or imputation is required.

#### Bias-Variance tradeoff for K
- **Problem with having too small K:** The major concern associated with small values of K lies behind the fact that the smaller value causes noise to have a higher influence on the result which will also lead to a large variance in the predictions. So, **choosing K to a small value may lead to a model with a large variance**.
- **Problem with having too large K:** The larger the value of K, the higher is the accuracy. However if K is too large, then our model is under-fitted. As a result, the error will go up again. So, to prevent your model from under-fitting it should retain the generalization capabilities otherwise there are fair chances that your model may perform well in the training data but drastically fail in the real data. The computational expense of the algorithm also increases if we choose the k very large. So, **choosing K to a large value may lead to a model with a large bias(error)**.

The effects of k values on the bias and variance is explained below:
- As the value of k increases, the bias will increase.
- As the value of k decreases, the variance will increase.
- With the increasing value of K, the boundary becomes smoother.

So, there is a tradeoff between overfitting (highly susceptible to individual training data points) and underfitting (not every susceptible to individual data points) and you have to maintain a balance while choosing the value of K in KNN. Therefore, K should not be too small or too large.

#### How to choose value of K?
K is a crucial parameter in the KNN algorithm. The optimum value of K for KNN is highly dependent on the data itself. There is no straightforward method to find the optimal value of K in the KNN algorithm. In different scenarios, the optimum K may vary. It is more or less a hit and trial method. Some suggestions for choosing K are:
- **Square Root Method:** Take the square root of the number of samples in the training dataset and assign it to the K value. This is a heurestic and is generally a good start, but there is no reason or explanation for this to be always True.
- **Cross-Validation Method**: This method involves starting from minimum value of K ie. K=1, and run cross-validation, measure the accuracy, and keep repeating till the results become consistent. As the value of K increases, the error usually goes down after each one-step increase in K, then stabilizes, and then raises again. Finally, pick the optimum K at the beginning of the stable zone. This technique is also known as the Elbow Method.

  **DOs and DONTs:**
  1. K should be the square root of n (number of data points in the training dataset).
  2. Always choose an odd value of K beacause in order to ensure that there are no ties in the voting.


#### Possible distance metrics that can be used:
- **Minkowski distance**: It is a metric intended for real-valued vector spaces. We can calculate Minkowski distance only in a normed vector space, which means in a space where distances can be represented as a vector that has a length and the lengths cannot be negative. Formula -> `[Σ(|Xi - Yi|)^p]^(1/p)`. The p value in the formula can be manipulated to give us different distances like Manhattan distance (p=1) and Euclidean distance (p=2).
- **Manhattan distance**: `Σ(|Xi - Yi|)`
- **Euclidean distance**: `[Σ(|Xi - Yi|)^2]^(1/2)`
- **Cosine Distance**: This distance metric is used mainly to calculate similarity between two vectors. It is measured by the cosine of the angle between two vectors and determines whether two vectors are pointing in the same direction. Formula -> `cos Θ = (a.b)/(|a|.|b|)`. Using this distance we get values between 0 and 1, where 0 means the vectors are 100% similar to each other and 1 means they are not similar at all.
- **Hamming Distance**: Hamming Distance measures the similarity between two strings of the same length. The Hamming Distance between two strings of the same length is the number of positions at which the corresponding characters are different. Hamming distance only works when we have strings or arrays of the same length. Example: distance between two strings: "euclidean" and "manhattan" is 7. **Integers or numbers can also be compared using hamming distance after conversion to binary strings.**

#### Advantages of KNN

-   **No Training Period**
-   **Seamless addition of new data:** Because the KNN algorithm does not require any training before making predictions as a result new data can be added seamlessly without impacting the accuracy of the algorithm.
-   **Easy to implement and understand**: To implement the KNN algorithm, we need only two parameters i.e. the value of K and the distance metric(e.g. Euclidean or Manhattan, etc.). Since both the parameters are easily interpretable therefore they are easy to understand

#### Downsides of KNN
-   **Does not consider closeness of data-points**: One of the limitations of KNN is that it doesn’t consider the closeness of data points from the target point i.e. a point that is very close will be considered as important as a point that is very far away (as long as it is within top K nearest points). One could look at Distance KNN - weighted average of values in regression/ weighted voting in classification, where weight is inversely proportional to distance (generally Euclidean measure).
-   **Does not work well with large datasets**: In large datasets, the cost of calculating the distance between the new point and each existing point is huge which decreases the performance of the algorithm.
-   **Does not work well with high dimensions**: Two data points need to be closer in every dimension such that the distance between them remains small and even if the distance is large just on a scale, it can impact the overall distance by a good magnitude.
-   **Sensitive to Noise and Outliers**: KNN is highly sensitive to the noise present in the dataset (especially when value of K is low) and requires manual imputation of the missing values along with outliers removal.
-   **High Inference time and memory needs**: Although KNN algorithm saves training time (as it need not be trained), the inference time for this algorithm is really high and scales with number of dimensions in data and number of samples. Also, we need to have access to training data all the time, which can be highly expensive based on dataset size.

#### Implementation of KNN from scratch

```python

import numpy as np

class KNN:
    def __init__(self, K):
        # hyper-parameters
        self.K = K

    def L1_distance(self, a, b):
        return np.absolute(a-b)

    def predict(self, x, X_train, Y_train):
        '''
        This code is written for binary classification problem
        with classes (0 and 1). X_train, Y_train represent the dataset.
        '''
        # in_shape -> [(N, ), (N, )], out_shape -> (N, )
        distance = self.L1_distance(x, X_train)
        # in_shape -> [(N, ), (N, )], out_shape -> (N, 2)
        distance_and_labels = np.stack((distance, Y_train), axis=1)

        sorted_labels_and_distances = distance_and_labels[np.argsort(distance_and_labels[:, 0])]
        smallest_K_distances = sorted_labels_and_distances[:self.K, 1]
        count_ones = sum(smallest_K_distances)
        count_zeros = self.K - count_ones

        return 0 if count_zeros > count_ones else 1
```

#### Interview questions
<!-- ----------------------------------------------------------------------------------------------- -->

**Q.** What is space and time complexity of the KNN Algorithm?  
**A.** **Time complexity:** The complexity of KNN is given by: **O(Nd + kN)** where N refers to number of samples in training data, **d** refers to dimension of features,  refers to **k** nearest neighbours. Reason being - calculating distances **O(d)** from target point for all points **O(N)**, and then calculating nearest neighbour **O(N)**, **k** times **O(k)**. If you assume N >> K, then complexity simplifies **O(N)**. This time complexity highly depends on the implementation and there are faster implementations of KNN using advanced data structures like KD-Trees. The memory of these algorithms vary too and also plays an important role in deciding the final implementation.
**Space complexity**: Since it stores all the pairwise distances and is sorted in memory on a machine, memory is also the problem. Usually, local machines will crash, if we have very large datasets.

<!-- ----------------------------------------------------------------------------------------------- -->

**Q.** KNN algoirthm does not consider closeness of data-points. How can you solve this?
**A.** One of the limitations of KNN is that it doesn’t consider the closeness of data points from the target point i.e. a point that is very close will be considered as important as a point that is very far away (as long as it is within top K nearest points). **One could look at Distance KNN - weighted average of values in regression/ weighted voting in classification, where weight is inversely proportional to distance (generally Euclidean measure).**

<!-- ----------------------------------------------------------------------------------------------- -->

**Q.** Is Feature Scaling required for the KNN Algorithm? Explain with proper justification.  
**A.** Yes, feature scaling is required to get the better performance of the KNN algorithm.
For Example, Imagine a dataset having n number of instances and N number of features. There is one feature having values ranging between 0 and 1. Meanwhile, there is also a feature that varies from -999 to 999. When these values are substituted in the formula of Euclidean Distance, this will affect the performance by giving higher weightage to variables having a higher magnitude.

<!-- ----------------------------------------------------------------------------------------------- -->

**Q.** Which algorithm can be used for value imputation in both categorical and continuous categories of data?  
**A.** KNN is the only algorithm that can be used for the imputation of both categorical and continuous variables. It can be used as one of many techniques when it comes to handling missing values. To impute a new sample, we determine the samples in the training set “nearest” to the new sample and averages the nearby points to impute.

<!-- ----------------------------------------------------------------------------------------------- -->

**Q.** Is it possible to use the KNN algorithm for Image processing?  
**A.** Yes, by converting a 3-dimensional image into a single-dimensional vector and then using it as the input to the KNN algorithm.

<!-- ----------------------------------------------------------------------------------------------- -->

**Q.** What are the real-life applications of KNN Algorithms?  
**A.** The various real-life applications of the KNN Algorithm includes:
-   KNN allows the calculation of the **credit rating**. By collecting the financial characteristics vs. comparing people having similar financial features to a database we can calculate the same. **Moreover, the very nature of a credit rating where people who have similar financial details would be given similar credit ratings also plays an important role**. Hence the existing database can then be used to predict a new customer’s credit rating, without having to perform all the calculations.
-   In political science, KNN can also be used to predict whether a potential voter “will vote” or “will not vote”, or to “vote Democrat” or “vote Republican” in an election

<!-- [Improve] -->

## What can we improve here?
- I am so lonely in here. This article cant be this perfect. #cringe_max :P

## References

1.  [Rajat Gupta ML Notes](https://ml-notes-rajatgupta.notion.site/)
2. [Analytics Vidhya - 20 questions to test your skills on KNN Algorithm](https://www.analyticsvidhya.com/blog/2021/05/20-questions-to-test-your-skills-on-k-nearest-neighbour/)
