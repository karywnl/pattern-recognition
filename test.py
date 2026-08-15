from src import linalg, imaging, plotting
import numpy as np
import sys

img = imaging.load_image("docs/assignments/lab-2-img/cat_02_square.png") 
img = imaging.resize(img, (100, 100))
A = imaging.to_grayscale(img)

print(A.min(), A.max())

k = 100

eigenvalues, Q = linalg.evd(A)
print(eigenvalues, len(list(eigenvalues)))

a_k_evd = linalg.reconstruct_evd(eigenvalues, Q, k)
reconstructed_eigen_values, reconstructed_Q = linalg.evd(a_k_evd)

count = 0
for i in a_k_evd.flatten():
    if abs(np.conj(i)) > 1e-8: count += 1

print(f"{count}/{a_k_evd.flatten().shape[0]}")
    

# print(reconstructed_eigen_values, len(list(reconstructed_eigen_values)))
plotting.show_reconstruction(A, a_k_evd, title=f"square EVD, k={k}")

n = len(eigenvalues)

if 0 < k < n and np.isclose(eigenvalues[k], np.conj(eigenvalues[k - 1])):
    k += 1

print()
print(reconstructed_eigen_values[:k])
