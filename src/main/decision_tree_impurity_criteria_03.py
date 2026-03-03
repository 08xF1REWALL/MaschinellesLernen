from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from pydotplus import graph_from_dot_data
from sklearn.tree import export_graphviz
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from mlxtend.plotting import plot_decision_regions

import matplotlib.pyplot as plt
import numpy as np

def gini(p):
    """Calculate the Gini impurity for a binary classification. Gini=1−(p2+(1−p)2)"""
    return (p) * (1 - (p)) + (1 - p) * (1 - (1 - p))
    

def entropy(p):
    """"Calculate the Entropy for a binary classification. Entropy=−plog2(p)−(1−p)log2(1−p)"""
    return - p*np.log2(p) - (1 - p)*np.log2(1 - p)

def error(p):
    """Calculate the Classification Error for a binary classification. Error=1−max(p,1−p)"""
    return 1 - np.maximum(p, 1 - p)

# Load and prepare data
iris = load_iris()
X = iris.data[:, [2, 3]]  # Use petal length and petal width
y = iris.target
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Standardize features
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# Train decision tree classifier
tree = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=1)
tree.fit(x_train, y_train)

# Plot decision regions
x_combined = np.vstack((x_train, x_test))
y_combined = np.hstack((y_train, y_test))
plot_decision_regions(X=x_combined, 
                     y=y_combined,
                     clf=tree)
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.title('Decision Tree Classifier')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

# Plot impurity criteria
x = np.arange(0.0, 1.0, 0.01)
ent = [entropy(p) if p != 0 else None for p in x]
sc_ent = [e*0.5 if e else None for e in ent]
err = [error(i) for i in x]

fig = plt.figure()
ax = plt.subplot(111)
for i, lab, ls, c in zip([ent, sc_ent, gini(x), err], 
                          ['Entropy', 'Entropy (scaled)', 
                           'Gini Impurity', 'Misclassification Error'],
                          ['-', '-', '--', '-.'],
                          ['black', 'lightgray', 'red', 'green']):
    line = ax.plot(x, i, label=lab, linestyle=ls, lw=2, color=c)

ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
          ncol=5, fancybox=True, shadow=False)

ax.axhline(y=0.5, linewidth=1, color='k', linestyle='--')
ax.axhline(y=1.0, linewidth=1, color='k', linestyle='--')
plt.ylim([0, 1.1])
plt.xlabel('p(i=1)')
plt.ylabel('Impurity Index')
#plt.savefig('images/03_19.png', dpi=300, bbox_inches='tight')
plt.show()

dot_data = export_graphviz(tree,
                           rounded=True,
                           class_names=['Setosa', 'Versicolor', 'Virginica'],
                            feature_names=['petal length', 'petal width'],
                            out_file=None)





forst = RandomForestClassifier(criterion='gini',
                              n_estimators=25,
                              random_state=1,
                              n_jobs=2)
forst.fit(x_train, y_train)
plot_decision_regions(X=x_combined, 
                     y=y_combined,
                     clf=forst)
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.title('Random Forest Classifier')
plt.legend(loc='upper left')
plt.show()


# KNN implementation
knn = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
knn.fit(x_train, y_train)
plot_decision_regions(X=x_combined, 
                     y=y_combined,
                     clf=knn)
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.title('K-Nearest Neighbors Classifier')
plt.show()

###################### Baum in dot-Format speichern ######################
# Create images directory if it doesn't exist
import os
os.makedirs('images', exist_ok=True)

# Save DOT file
with open('images/decision_tree.dot', 'w') as f:
    f.write(dot_data)
print("Decision tree DOT file saved to 'images/decision_tree.dot'")

# Try to save as PNG using pydotplus
try:
    graph = graph_from_dot_data(dot_data)
    graph.write_png('images/decision_tree.png')
    print("Decision tree PNG saved to 'images/decision_tree.png'")
except Exception as e:
    print(f"Could not generate PNG: {e}")
    print("\nTo visualize the tree, use one of these options:")
    print("1. Install GraphViz: https://graphviz.org/download/")
    print("2. Use online viewer: http://www.webgraphviz.com/")
    print("3. View the DOT file: images/decision_tree.dot")

