# Amir Shahbazi - 9812033

import numpy as np
import math


def get_matrix_A():
    print("Insert your numbers for matrix 'A' in a line with space between them:)")
    matrix_A = input().split()
    matrix_A = list(map(float, matrix_A))
    l = int(math.sqrt(len(matrix_A)))
    matrix_A = np.array(matrix_A).reshape(l, l)
    return matrix_A


def get_matrix_b():
    print("Insert your numbers for matrix 'b' in a line with space between them:)")
    matrix_b = input().split()
    matrix_b = list(map(float, matrix_b))
    l = len(matrix_b)
    matrix_b = np.array(matrix_b).reshape(l, 1)
    return matrix_b


def display(M, n):
    for i in range(n):
        for j in range(n):
            print(M[i][j], end="\t")
        print("\n")


def positive_definite(M):
    l = M.shape[0]
    test = 0
    for i in range(l):
        a = M[0: i + 1, 0: i + 1]
        if np.linalg.det(a) > 0:
            test += 1
        else:
            break
    if test == l:
        print("Positive Definite : Yes")
        print("\n")
        return True
    else:
        print("Positive Definite : No")
        print("\n")
        return False


def cholesky(M, n):
    L = np.zeros(n * n).reshape(n, n)
    for i in range(n):
        for j in range(i + 1):
            s = 0
            if j == i:
                for k in range(j):
                    s += pow(L[j, k], 2)
                L[j, j] = math.sqrt(M[j, j] - s)
            else:
                for k in range(j):
                    s += L[i, k] * L[j, k]
                if L[j, j] > 0:
                    L[i, j] = (M[i, j] - s) / L[j, j]
    L_T = np.transpose(L)
    return (L, L_T)


def LU(M, n):
    L = np.identity(n)
    for i in range(n):
        I = np.identity(n)
        x = M[i, i]
        for j in range(i + 1, n):
            r = -(M[i, j] / x)
            I[j, i] = r
            M[j] = r * M[i] + M[j]
        L = np.dot(L, np.linalg.inv(I))
    U = M
    return (L, U)


def decomposition(M):
    n = len(M[0])
    if positive_definite(M):
        (L, L_T) = cholesky(M, n)
        print("Cholesky decomposition :")
        print("L :")
        display(L, n)
        print("L(Transpose) :")
        display(L_T, n)
        return (L, L_T)
    else:
        (L, U) = LU(M, n)
        print("LU decomposition :")
        print("L :")
        display(L, n)
        print("U :")
        display(U, n)
        return (L, U)


def sovle(A, b):
    (L, U) = decomposition(A)
    Y = np.linalg.solve(L, b)
    X = np.linalg.solve(U, Y)
    print("Final Answere: ")
    print("X :")
    print(X)


A = get_matrix_A()
b = get_matrix_b()
sovle(A, b)
