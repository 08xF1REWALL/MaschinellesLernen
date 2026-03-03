import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np 
import pandas as pd

# ────────────────────────────────────────────────────────────────
# Your Perceptron class (unchanged except small typo fix in name)
# ────────────────────────────────────────────────────────────────

class Perceptron(object):           # ← fixed typo: Preceptron → Perceptron
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta # learning rate
        self.n_iter = n_iter # number of epochs
        self.random_state = random_state
    
    def fit(self, X, y):
        # random number generator
        rgen = np.random.RandomState(self.random_state)
        # weights initialization with small random numbers
        
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.errors_ = []
        for _ in range(self.n_iter):
            error = 0
            for xi, target in zip(X, y):
                # Δw = η (y - ŷ) x
                # ← HIER STARTET DIE REGEL: Innere Schleife über jedes Sample x^{(i)} und y^{(i)}
                # Schritt 1: Berechne ŷ (output) für das aktuelle Sample xi
                # (self.predict(xi) ruft intern net_input auf und wendet die Step-Funktion an)
                update = self.eta * (target - self.predict(xi))# ← Berechnung des Fehlers und des Updates
                self.w_[1:] += update * xi
                self.w_[0] += update
                error += int(update != 0.0)
            self.errors_.append(error)
        return self
    
    def net_input(self, X):
        # calculate net input
        # Berechnet z=w0+∑j=1 wj⋅xj^(i)
        # # x₁w₁ + x₂w₂ + ... + bias
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def predict(self, X):
        # return class label after unit step
        # Step-Funktion: Wenn z >= 0, dann +1, sonst -1

        return np.where(self.net_input(X) >= 0.0, 1, -1)


# ────────────────────────────────────────────────────────────────
# FIXED version - moved OUTSIDE the class + added self parameter removed
# ────────────────────────────────────────────────────────────────

def plot_decision_regions(X, y, classifier, resolution=0.02):
    """
    Plots decision regions of a classifier for 2D feature space.
    
    Parameters:
    -----------
    X : array-like, shape = [n_examples, n_features]
        Training vectors
    y : array-like, shape = [n_examples]
        Target values
    classifier : object
        Fitted classifier with .predict() method
    resolution : float
        Resolution of the decision boundary grid
    """
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    
    # make predictions for all points in the grid
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.8, 
                    c=colors[idx],
                    marker=markers[idx], 
                    label=cl,
                    edgecolor='black')


# ────────────────────────────────────────────────────────────────
# load iris data from UCI repository
# ────────────────────────────────────────────────────────────────

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
# print(df.tail())
# select setosa and versicolor, setosa is -1, versicolor is 1 
y = df.iloc[0:100,  4].values
y = np.where(y == 'Iris-setosa', -1, 1) # text if setosa then -1 else 1
# extract sepal length and petal length, column 0 and 2, first 100 rows
X = df.iloc[0:100, [0, 2]].values   
# Decision boundary
#x1 = np.linspace(X[:,0].min()-0.5, X[:,0].max()+0.5, 200)
#x2 = -x1 + 5
#plt.plot(x1, x2, linestyle='--', linewidth=2, label='decision boundary')

plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa') # column 0 and 1, 0 is x and 1 is y
plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor') # row 50 to 100 0 is sepal length, 1 is petal length 
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')
plt.show()


# ────────────────────────────────────────────────────────────────
# Training + plots
# ────────────────────────────────────────────────────────────────

ppn = Perceptron(eta=0.1, n_iter=10, random_state=1)
ppn.fit(X, y)

plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of updates')
plt.show()

plt.figure(figsize=(7, 5))
plot_decision_regions(X, y, classifier=ppn)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend(loc='upper left')
plt.title('Decision Regions')
plt.show()
