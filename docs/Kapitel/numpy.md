# NumPy documentation
1. np.where : Return elements chosen from x or y depending on condition.
```py
    np.where(condition, x, y)
    # Example
    a = np.array([1, 2, 3, 4, 5])
    np.where(a > 3, a, -1)
    # Output: array([-1, -1, -1,  4,  5])
```
2. np.linspace : Return evenly spaced numbers over a specified interval.
```py
    np.linspace(start, stop, num=50)
    # Example
    np.linspace(0, 1, 5)
    # Output: array([0.  , 0.25, 0.5 , 0.75, 1.  ])
```
3. np.dot: Dot product of two arrays.
```py
    np.dot(a, b)
    # Example
    a = np.array([1, 2])
    b = np.array([3, 4])
    np.dot(a, b)
    # Output: 11
```
4. np.where: Returns elements from x or y depending on the condition.

5. np.meshgrid : Create coordinate matrices from coordinate vectors.
```py
    np.meshgrid(x, y)
    # Example
    x = np.array([1, 2, 3])
    y = np.array([4, 5])
    X, Y = np.meshgrid(x, y)
    # Output:
    # X = array([[1, 2, 3],
    #            [1, 2, 3]])
    # Y = array([[4, 4, 4],
    #            [5, 5, 5]])
```

6. np.arange: Return evenly spaced values within a given interval.
```py
    np.arange(start, stop, step)
    # Example
    np.arange(0, 10, 2)
    # Output: array([0, 2, 4, 6, 8])
```
7. xx1.ravel(): Return a contiguous flattened array.
```py
    xx1 = np.array([[1, 2], [3, 4]])
    xx1.ravel()
    # Output: array([1, 2, 3, 4])
```

8. np.unique: Find the unique elements of an array.
```py
    np.unique(ar)
    # Example
    ar = np.array([1, 2, 2, 3, 4, 4, 5])
    np.unique(ar)
    # Output: array([1, 2, 3, 4, 5])
```

9. reshape : Gives a new shape to an array without changing its data.
```py
    a.reshape(newshape)
    # Example
    a = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    a.reshape(4, 2)
    # Output: array([[1, 2],
    #                [3, 4],
    #                [5, 6],
    #                [7, 8]])
```

10. np.vstack: Stack arrays in sequence vertically (row wise).
```py
    np.vstack(tup)
    # Example
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    np.vstack((a, b))
    # Output: array([[1, 2, 3],
    #                [4, 5, 6]])
```
