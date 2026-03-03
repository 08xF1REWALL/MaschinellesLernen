# A Tour of Machine Learning Classifiers Using scikit-learn

 The five main steps that are involved in training machine learning algorithms can be summarized as follows:
 1. Selecting features and collecting training data
 2. Choosing a performance metric
 3. Choosing a classifier and optimizing algorithm.
 4. Evaluating the preformance of the model.
 5. Tuning the algorithm. 

## First steps with scikit-learn- training a preceptron
scikit-learn api: library offers a large variety of machine learning algorithms, also alot of functions for data preprocessing, model evaluation and hyperparameter tuning.

```py
from sklearn import datasets
import numpy as np


iris = datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two features.
y = iris.target # corresponding class labels of the flowers
print('Class labels:', np.unique(y))
```

```py
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) # split data into training and test set, 30% test data samples, 70% training data samples 
```

- stratification: means that the train_test_split method returns training and test subsets that have the same proportions of class labels as the input dataset.
We can use Numpys bincount function to countss the number of occurrences of each value in an array, to verify that his is indeed the case:

```py
print('Labels counts in y:', np.bincount(y))
print('Labels counts in y_train:', np.bincount(y_train))
print('Labels counts in y_test:', np.bincount(y_test))
```
- feature scaling: in the gradient descent example. Here we will standardize the features using the standardscaler class from scikit-learns preprocessing module.

```py
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)  
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
```

- calculate the classification accuracy as follows:

```py
from sklearn.metrics import accuracy_score
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
```

- Modeling class probabilites via logistic regression:

the biggest disadvantage of the preceptron is that it never converges if the classes are not perfectly linearly separable. 

## Logistic regression intuition and conditional probabilities:

it's a classification model that is very easys to implement but preforms very well on liniarly separable classes "linear trennbare Klassen.".

- the odds ratio_ the odds in favor of a particular event. The odds ratio can be written as p/(1-p), where p is the probability of the positon event. 

- logit function: the natural logarithm of the odds ratio, i.e., log(p/(1-p)). The logit function maps probabilities from the interval (0, 1) to the entire real number line (-∞, +∞).
log refers to the natural logarithm. The logit function takes as input values in range of 0 to 1 and transforms them to values in the range of -∞ to +∞. which we can use to express a linear relationship between feature values and the log-odds

logit(p(y=1|x)) = w0x0 + w1x1 + ... + wmxm= ∑i=0m wixi= wT x

here, p(y=1|x) is the conditional probability that a particular sample belongs to class 1 given its features x.

- sigmoid function 
Now we are interested in the probability that a certain sample belogs to a particular class. Which is the inverse form of the logit function. It is called the logistic function or sigmoid 

function:

sigmoid(z) = 1/(1+exp(-z))
here z is the net input, the linear combination of the weights and sample features
z = wT x =  w0x0 + w1x1 + ... + wmxm

the output of the sigmoid function is then interpreted as the probability of a particular sample beloging to class 1, σ(z) = p(y=1|x;w), given its features x parameterized by the weights w. For example, if we compute σ(z) = 0.8 for a particular sample, we can interpret this as an 80% probability that the sample belongs to class 1. Therefore the probability that this flower is an iris-setosa flower can be calculated as p(y=0|x;w) = 1 - p(y=1|x;w) = 1 - σ(z) = 1-0.8a = 0.2, or 20%.


## Training a logistic regression model with scikit-learn

## Dealing with nonlinearly separable case using slack variables

```py
from sklearn.svm import SVC
svm = SVC(kernel = 'linear', C=1.0, random_state=1)
svm.fit(X_train_std, y_train)
plot_decision_regions(X_combined_std, y_combined, classifier=svm, test_idx=range(105,150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.show()
```

## Solving nonlinear problems using a kernel SVM
kernal methods for linearly inseparable data

```py
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(1)
X_xor = np.random.randn(200, 2)
y_xor = np.logical_xor(X_xor[:, 0] > 0, X_xor[:, 1] > 0)
y_xor = np.where(y_xor, 1, -1)
plt.scatter(X_xor[y_xor == 1, 0], X_xor[y_xor == 1, 1],
            c='b', marker='x', label='1')
plt.scatter(X_xor[y_xor == -1, 0], X_xor[y_xor == -1, 1],
            c='r', marker='s', label='-1')
plt.xlim(-3, 3)
plt.ylim(-3, 3)
plt.legend(loc='best')
plt.show()
```

## Decision tree learning
```py
```


## Combining multiple decision trees via random forests
The random forest can be considered as an ensemble of decisioin trees. The idea behind a random forest is to average multiple(deep) decision trees that individually suffer from high variance, to build a more robust model that has a better generalization preformance and is less susceptible to overfitting. 
this can be summarized as follows:
1. Draw a random bootstrap sample of size n(randomly choose n samples from the training set with replacement)
2. Grow a decision tree from the bootstrap sample. At each node:
    a. Randomly select d features without replacement.
    b. Split the node using the features that provides the best split accourding to the objective function, for instance, maximizing the information gain.
3. Repeat the steps 1-2 k times.
4. Aggregate the prediction by each tree to assign the class label by majority vote.  
- good classification preformance, scalability, and easy to use. 

## K-Nearest Neighbors (KNN) - a lazy learning algorithm
this doesn't learn a discrominative function from the training data, but memorizes the training dataset instead.
KNN workwise:
1. choose the number of k and a distance metric.
2. find the k-nearest neighbors of the sample that we want to classify
3. assign the class label by majority vote.
