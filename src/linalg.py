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


# sort eigenvalues by descending magnitude while keeping each eigenvector paired
# with its corresponding eigenvalue
def sort_eigenpairs(eigenvalues, Q):
    eigenvalues = eigenvalues.copy()
    Q = Q.copy()
    n = len(eigenvalues)

    for i in range(n):
        largest = i

        for j in range(i + 1, n):
            if np.abs(eigenvalues[j]) > np.abs(eigenvalues[largest]):
                largest = j

        if largest != i:
            temp_value = eigenvalues[i]
            eigenvalues[i] = eigenvalues[largest]
            eigenvalues[largest] = temp_value

            temp_vector = Q[:, i].copy()
            Q[:, i] = Q[:, largest]
            Q[:, largest] = temp_vector

    return eigenvalues, Q


# eigendecomposition, sorted by descending eigenvalue magnitude
def evd(A):
    eigenvalues, Q = np.linalg.eig(A)
    return sort_eigenpairs(eigenvalues, Q)

# rank actually used after keeping conjugate pairs together
def effective_k(eigenvalues, k):
    n = len(eigenvalues)
    if 0 < k < n and np.isclose(eigenvalues[k], np.conj(eigenvalues[k - 1])):
        k += 1
    return k

# rank-k reconstruction A_k = Q * lambda_k * Q^-1
def reconstruct_evd(eigenvalues, Q, k):
    n = len(eigenvalues)
    k = effective_k(eigenvalues, k)

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
