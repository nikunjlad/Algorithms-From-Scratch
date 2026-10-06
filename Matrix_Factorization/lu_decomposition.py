"""
LU Decomposition Algorithm

This code helps to understand LU decomposition.
LU decomposition is used internally to factorize large matrix multiplication to save computation time
It helps to solve a system of linear equations

For instance Ax=b
where A=3x3 matrix, x=3x1 vector and b=3x1 vector

what LU does is decompose A into Lower triagular (L) and Upper triangular (U)
Gaussian eliminiation helps achieve U and L matrices
L = [[1  ,0  ,  0],
     [L10,1  ,  0],
     [L20,L21,  1]]

U = [[U00, U01, U02],
     [0  , U11, U12],
     [0  , 0  , U22]]

Since A = LU, our equation becomes -> LUx=b
where we say -> Ux=y  (Backward Substituition)
             -> Ly=b  (Forward Substituition)

Steps for LU Decomposition
1. Decompose A into L and U matrices using Gaussian Elimination
2. Perform forward substituition and get y using L and b matrices
3. Perform backward substituition and get x using U and y matrices
"""

import numpy as np


def lu_decomposition(A):

    n = len(A)      # get length of matrix A (number of rows)
    L = np.eye(n)   # create L as an identity matrix
    U = A.copy()    # create U as a copy of A

    for i in range(n):
        for j in range(i+1,n):
            factor = U[j][i] / U[i][i]
            L[j][i] = factor
            U[j,i:] -= factor * U[i,i:]

    return L, U

def forward_substituition(L,b):

    n = len(b)
    y = np.zeros_like(b)

    for i in range(n):
        y[i] = b[i] - L[i,:i].dot(y[:i])

    return y

def backward_substitution(U,x):

    n = len(y)
    x = np.zeros_like(x)

    for i in range(n-1,-1,-1):
        x[i] = (y[i] - U[i,i+1:].dot(x[i+1:])) / U[i,i]

    return x

A = np.array([[4.0,-2.0,1.0],
              [-3.0,-1.0,4.0],
              [1.0,-1.0,3.0]])
b = np.array([15.0,8.0,13.0])
L,U = lu_decomposition(A)
print(f"Matrix A: {A}")
print(f"Matrix L: {L}")
print(f"Matrix U: {U}")

y = forward_substituition(L,b)
print(f"y: {y}")

x = backward_substitution(U,y)
print(f"x: {x}")
