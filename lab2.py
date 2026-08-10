from src import imaging, linalg, plotting

ks = [5, 20, 50]

# Task 1: square image - EVD + SVD
img = imaging.load_image("docs/assignments/lab-2-img/cat_02_square.png")
img = imaging.resize(img, (100, 100))
A = imaging.to_grayscale(img)

eigenvalues, Q = linalg.evd(A)
U, s, Vt = linalg.svd(A)

for k in ks:
    a_k_evd = linalg.reconstruct_evd(eigenvalues, Q, k)
    a_k_svd = linalg.reconstruct_svd(U, s, Vt, k)
    print(f"square k={k}  E_evd={linalg.frobenius_norm(A - a_k_evd):.2f}  E_svd={linalg.frobenius_norm(A - a_k_svd):.2f}")
    plotting.show_reconstruction(A, a_k_evd, title=f"square EVD, k={k}")
    plotting.show_reconstruction(A, a_k_svd, title=f"square SVD, k={k}")

n = len(eigenvalues)
all_ks = list(range(1, n + 1))
errors_evd = [linalg.frobenius_norm(A - linalg.reconstruct_evd(eigenvalues, Q, k)) for k in all_ks]
errors_svd = [linalg.frobenius_norm(A - linalg.reconstruct_svd(U, s, Vt, k)) for k in all_ks]
plotting.error_curve(all_ks, {"evd": errors_evd, "svd": errors_svd}, title="square image: E(k) vs k")

# Task 2: rectangular image - SVD only
img2 = imaging.load_image("docs/assignments/lab-2-img/cat_02_rectangle.png")
img2 = imaging.resize_preserve_aspect(img2, 100)
A2 = imaging.to_grayscale(img2)

U2, s2, Vt2 = linalg.svd(A2)

for k in ks:
    a_k = linalg.reconstruct_svd(U2, s2, Vt2, k)
    print(f"rectangle k={k}  E_svd={linalg.frobenius_norm(A2 - a_k):.2f}")
    plotting.show_reconstruction(A2, a_k, title=f"rectangle SVD, k={k}")

n2 = len(s2)
all_ks2 = list(range(1, n2 + 1))
errors2 = [linalg.frobenius_norm(A2 - linalg.reconstruct_svd(U2, s2, Vt2, k)) for k in all_ks2]
plotting.error_curve(all_ks2, {"svd": errors2}, title="rectangle image: E(k) vs k")
