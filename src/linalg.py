import numpy as np

# dot product
def dot(a, b):
    tot = 0
    for i in range(len(a)):
        tot += a[i] * b[i]
    return tot    

# transpose mxn to nxm
def transpose(A):
    m = len(A)
    n = len(A[0])

    arr = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            arr[i, j] = A[j][i]

    return arr

# matrix multiplication
def matmul(A, B):
    # A (mxn) * B (nxk) = C (mxk)
    m = len(A)
    n = len(B)
    k = len(B[0])

    B_t = transpose(B)
    arr = np.zeros((m, k))

    for i in range(m):
        for j in range(k):
            arr[i, j] = dot(A[i], B_t[j])

    return arr


# is symmetric matrix
def is_symmetric(A):
    m = len(A)
    n = len(A[0])

    if m != n:
        return False
    
    for i in range(m):
        for j in range(i+1, n):
            if A[i][j] != A[j][i]:
                return False

    return True



