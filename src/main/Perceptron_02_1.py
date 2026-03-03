import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# ────────────────────────────────────────────────────────────────
# Perceptron class (fixed spelling)
# ────────────────────────────────────────────────────────────────
class Perceptron(object):
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
    
    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.errors_ = []
        for _ in range(self.n_iter):
            error = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                error += int(update != 0.0)
            self.errors_.append(error)
        return self
    
    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)


# ────────────────────────────────────────────────────────────────
# Decision regions plotting function (standalone!)
# ────────────────────────────────────────────────────────────────
def plot_decision_regions(X, y, classifier, resolution=0.02):
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.8, 
                    c=colors[idx],
                    marker=markers[idx], 
                    label=cl, 
                    edgecolor='black')


# ────────────────────────────────────────────────────────────────
# UNCOMMENTED EXAMPLE - Now it will run and show the plot!
# ────────────────────────────────────────────────────────────────

# Load Iris dataset directly from URL (only setosa & versicolor)
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
                 header=None, encoding='utf-8')

# Take first 100 rows (setosa = 0-49, versicolor = 50-99)
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)           # labels: -1 and 1

# Features: sepal length (column 0) and petal length (column 2)
X = df.iloc[0:100, [0, 2]].values

# Create & train perceptron
ppn = Perceptron(eta=0.1, n_iter=10, random_state=1)
ppn.fit(X, y)

# Plot everything!
plt.figure(figsize=(8, 6))
plot_decision_regions(X, y, classifier=ppn)
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')
plt.title('Perceptron - Iris setosa vs. versicolor\n(Decision Regions after 10 epochs)')
plt.grid(True, alpha=0.3)
plt.show()