from src import linalg, distributions, plotting

# Part 1: matrix/vector operations
A = [[1, 2, 3],
     [4, 5, 6]]                 # 2x3

B = [[7, 8],
     [9, 10],
     [11, 12]]                  # 3x2

v1 = [1, 2, 3, 4]
v2 = [5, 6, 7, 8]

S = [[2, 1, 0],
     [1, 3, 1],
     [0, 1, 4]]                 # symmetric

U = [[1, 2, 3],
     [0, 4, 5],
     [0, 0, 6]]                 # upper triangular

L = [[1, 0, 0],
     [2, 3, 0],
     [4, 5, 6]]                 # lower triangular

print("dot(v1, v2):", linalg.dot(v1, v2))
print("transpose(A):\n", linalg.transpose(A))
print("matmul(A, B):\n", linalg.matmul(A, B))
print("is_symmetric(S):", linalg.is_symmetric(S))
print("is_symmetric(A) [non-square]:", linalg.is_symmetric(A))
print("is_upper(U):", linalg.is_upper(U))
print("is_lower(L):", linalg.is_lower(L))


# Part 2(a): uniform samples in [a, b)
uniform_samples = distributions.uniform(a=0, b=1, n=20000)
plotting.histogram(uniform_samples, title="Uniform Distribution", bins=10)

# Part 2(b)/(c): gaussian samples with N(mu, sigma^2)
gaussian_samples = distributions.gaussian(mu=0, sigma=1, n=20000)
plotting.histogram(gaussian_samples, title="Gaussian Distribution")
