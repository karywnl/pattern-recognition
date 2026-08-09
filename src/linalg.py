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

# is upper triangular matrix
def is_upper(A):
    m = len(A)
    n = len(A[0])

    if m != n:
        return False

    # selecting the i>j region
    for i in range(1, m):
        for j in range(i):
            if A[i][j] != 0:
                return False

    return True

# is lower triangular matrix
def is_lower(A):
    m = len(A)
    n = len(A[0])

    if m != n:
        return False

    # selecting the i<j region
    for i in range(m):
        for j in range(i+1, n):
            if A[i][j] != 0:
                return False

    return True


# eigendecomposition, sorted by descending eigenvalue magnitude
def evd(A):
    eigenvalues, Q = np.linalg.eig(A)
    order = np.argsort(np.abs(eigenvalues))[::-1]
    return eigenvalues[order], Q[:, order]

# rank-k reconstruction A_k = Q * lambda_k * Q^-1
def reconstruct_evd(eigenvalues, Q, k):
    n = len(eigenvalues)

    # keep conjugate pairs together
    if 0 < k < n and np.isclose(eigenvalues[k], np.conj(eigenvalues[k - 1])):
        k += 1

    lambda_k = np.zeros(n, dtype=complex)
    lambda_k[:k] = eigenvalues[:k]
    lambda_k = np.diag(lambda_k)

    q_inv = np.linalg.inv(Q)
    a_k = Q @ lambda_k @ q_inv
    return np.real(a_k)

# singular value decomposition, sorted descending by construction
def svd(A):
    U, singular_values, Vt = np.linalg.svd(A)
    return U, singular_values, Vt

# rank-k reconstruction A_k = U * sigma_k * V^t
def reconstruct_svd(U, singular_values, Vt, k):
    m = U.shape[0]
    n = Vt.shape[0]

    sigma_k = np.zeros((m, n))
    sigma_k[:k, :k] = np.diag(singular_values[:k])

    return U @ sigma_k @ Vt

# frobenius norm: sqrt of sum of squares of every entry
def frobenius_norm(A):
    return np.sqrt(np.sum(A ** 2))


