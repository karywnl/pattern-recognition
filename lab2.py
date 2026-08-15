from pathlib import Path

from src import imaging, linalg, plotting

ks = [5, 20, 50]
figures_dir = Path("docs/submissions/lab2/figures")
figures_dir.mkdir(parents=True, exist_ok=True)

# Task 1: square image - EVD + SVD
loaded_img = imaging.load_image("docs/assignments/lab-2-img/cat_02_square.png")
img = imaging.resize(loaded_img, (100, 100))
A = imaging.to_grayscale(img)

plotting.show_pipeline(
    [loaded_img, img, A],
    ["loaded (512x512)", "resized (100x100)", "grayscale matrix A (100x100)"],
    save_path=figures_dir / "square_pipeline.png",
)

eigenvalues, Q = linalg.evd(A)
U, s, Vt = linalg.svd(A)

for k in ks:
    a_k_evd = linalg.reconstruct_evd(eigenvalues, Q, k)
    a_k_svd = linalg.reconstruct_svd(U, s, Vt, k)
    print(f"square k={k}  E_evd={linalg.frobenius_norm(A - a_k_evd):.2f}  E_svd={linalg.frobenius_norm(A - a_k_svd):.2f}")
    plotting.show_reconstruction(
        A,
        a_k_evd,
        title=f"square EVD, k={k}",
        save_path=figures_dir / f"square_evd_{k}.png",
    )
    plotting.show_reconstruction(
        A,
        a_k_svd,
        title=f"square SVD, k={k}",
        save_path=figures_dir / f"square_svd_{k}.png",
    )

n = len(eigenvalues)
all_ks = list(range(1, n + 1))
# conjugate pairs are kept whole, so only these ranks are actually reachable
ks_evd = sorted({linalg.effective_k(eigenvalues, k) for k in all_ks})
errors_evd = [linalg.frobenius_norm(A - linalg.reconstruct_evd(eigenvalues, Q, k)) for k in ks_evd]
errors_svd = [linalg.frobenius_norm(A - linalg.reconstruct_svd(U, s, Vt, k)) for k in all_ks]
plotting.error_curve(
    {"evd": (ks_evd, errors_evd), "svd": (all_ks, errors_svd)},
    title="square image: E(k) vs k",
    save_path=figures_dir / "square_curve.png",
)

# Task 2: rectangular image - SVD only
loaded_img2 = imaging.load_image("docs/assignments/lab-2-img/cat_02_rectangle.png")
img2 = imaging.resize_preserve_aspect(loaded_img2, 100)
A2 = imaging.to_grayscale(img2)

plotting.show_pipeline(
    [loaded_img2, img2, A2],
    ["loaded (520x600)", "resized (87x100)", "grayscale matrix A (100x87)"],
    save_path=figures_dir / "rect_pipeline.png",
)

U2, s2, Vt2 = linalg.svd(A2)

for k in ks:
    a_k = linalg.reconstruct_svd(U2, s2, Vt2, k)
    print(f"rectangle k={k}  E_svd={linalg.frobenius_norm(A2 - a_k):.2f}")
    plotting.show_reconstruction(
        A2,
        a_k,
        title=f"rectangle SVD, k={k}",
        save_path=figures_dir / f"rect_svd_{k}.png",
    )

n2 = len(s2)
all_ks2 = list(range(1, n2 + 1))
errors2 = [linalg.frobenius_norm(A2 - linalg.reconstruct_svd(U2, s2, Vt2, k)) for k in all_ks2]
plotting.error_curve(
    {"svd": (all_ks2, errors2)},
    title="rectangle image: E(k) vs k",
    save_path=figures_dir / "rect_curve.png",
)
