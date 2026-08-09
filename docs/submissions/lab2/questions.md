# Lab 2 report — questions

Answer each in your own words, a few sentences is fine. Leave a blank line
under each question and write your answer there. Ping me when you're done
and I'll assemble the PDF from exactly what you've written here, plus the
figures/tables/numbers we already generated from running the code.

Title page fields (name, ID, department, date) — fill in whenever, doesn't
need to happen before the rest.

## Introduction

1. In 2-3 sentences, what did Lab 2 ask you to do overall — both the square-image
   task and the rectangular-image task? Summarize it like you're explaining it
   to someone who hasn't read the assignment.


This lab is all about dimensionality reduction. We use decomposition techniques such as eigen value decomposition (diagonalization) and singular value decomposition to decompose an image and try to reconstruct it with fewer dimensions than the original amount. The square image gets both EVD and SVD, while the rectangular image only gets SVD.


2. Why do EVD and SVD matter for this kind of task — what's the connection
   between decomposing a matrix and approximating/compressing an image?

Both these techniques do the similar thing, they decompose the matrix (image representation) into its eigen values and eigen vectors and then try to reconstruct the original image with as few of these eigen values and vectors as possible. Thus, making a compressed form of an image (approximate).

## Experiments Performed — Question 1 (square image: EVD + SVD)

3. Walk through, step by step, what your code does to go from the raw color
   image to the final grayscale matrix A (resizing, then the grayscale formula).

- Load : So, first we use matplotlib image to load the image.
- Resize : then we need to resize it to 100x100 if it's a square matrix, or if it's a rectangular image you preserve the aspect ratio by keeping the longer side to 100 and the smaller side adjusted accordingly.
- Grayscale : we need to convert them into grayscale, and that is taken care of by extracting the RGB and using an arithmetic formula (0.299 * R + 0.587 * G + 0.114 * B)

Then we perform the decomposition and try to reconstruct it using different values of k.


4. In your own words, what is EVD, and what do Q and Λ mean in A = QΛQ⁻¹?

Eigen value decomposition, also known as diagonalization. It's a process used to decompose a square matrix into a diagonal matrix Λ and its eigen basis. Here we decompose a matrix A into a diagonal matrix Λ containing the eigenvalues, and its eigen basis contained in the matrix Q. There are multiple reasons we do this, one of them being to study the matrix through its eigenvalues and eigenvectors.


5. In your own words, what is SVD, and what do U, Σ, V mean in A = UΣVᵗ?

Singular value decomposition (SVD). While diagonalization is helpful, it can only be performed on square matrices. SVD extends this idea to non-square matrices. We have two matrices, AA^T giving us some eigenvalues and corresponding eigenvectors stacked in U, and A^TA giving us some eigenvalues and corresponding eigenvectors stacked in V. Σ is the singular value matrix sharing the same dimensions as A, containing the singular values, and singular values are the square root of the eigenvalues. The basic idea is that a non-square matrix cannot be diagonalized directly, so AA^T and A^TA are used to make it square first.

6. You picked k = 5, 20, 50. Why these three — what's your reasoning for a
   "low / mid / high" spread relative to n = 100?

We jumped from a very low point (k=5, heavy compression) to k=50, which is half the image's dimension, gradually increasing k in between (k=20) to show the difference in errors across that range.

7. Describe what you actually see in the reconstructed and error images at
   each k, for both EVD and SVD (refer to the figures in the visual reference
   PDF).

SVD performs considerably better than EVD for even square images. This is because SVD's eigenvectors U and V are orthogonal no matter what A looks like, while EVD's Q is only orthogonal when A itself is symmetric — which it isn't in our case — so that difference in orthogonality is what's causing the gap.

8. Compare the E(k) numbers you got for EVD vs SVD at k = 5, 20, 50 — what
   pattern do you see?

| k | E(k) EVD | E(k) SVD | ratio |
|---|---|---|---|
| 5 | 3281.02 | 1736.93 | ~1.9x |
| 20 | 2575.44 | 581.84 | ~4.4x |
| 50 | 765.98 | 122.67 | ~6.2x |

SVD's error is consistently lower than EVD's at every k, and the gap actually widens as k grows (~1.9x at k=5 up to ~6.2x at k=50).

9. Why do you think SVD outperformed EVD at every k? (Think back to Q not
   being orthonormal, and the Eckart–Young argument — explain it your way.)

SVD guarantees orthogonality — its vectors are perpendicular to each other, so none of them overlap and each one is new information on its own. When you keep the important ones and drop the smaller, unwanted ones, the error visibly drops. That is why EVD's error fluctuates a little instead of gradually decreasing, while SVD's error gradually decreases.


10. Did complex eigenvalues show up in your EVD computation? Briefly explain
    why, for this kind of matrix, and how your code deals with them before the
    result becomes a displayable image.

Yeah, 82 out of 100 came out to be complex eigen values. This is because A isn't symmetric, so the characteristic polynomial's roots aren't guaranteed to be real, and it splits out complex numbers instead. In the code, this is handled by not splitting conjugate pairs across the k boundary when truncating, and then stripping out the imaginary part and keeping only the real part before the result becomes a displayable image.



## Experiments Performed — Question 2 (rectangular image: SVD only)

11. Why is only SVD possible here, not EVD?

EVD can only handle square matrices and it can't handle non-square matrices. That's the main reason why we have SVD in the first place.

12. Describe how the rectangular image was resized (aspect ratio preserved) —
    give the original and final dimensions.
   
from the image the width and height are extracted and after that the maximum size is fixed to 100 and another is resized to maintain the aspect ratio using (min_side * 100 / max_side).


13. Describe your observations for the rectangular reconstructions and E(k)
    curve at k = 5, 20, 50.

| k | E(k) SVD |
|---|---|
| 5 | 1100.14 |
| 20 | 251.75 |
| 50 | 34.15 |

The error drops gradually and smoothly as k increases, the same monotonic behaviour SVD showed on the square image.


## Inference

14. Looking back at the whole lab, what did you learn about the difference
    between EVD and SVD, and when you'd prefer one over the other?

If it's a square matrix and symmetric already just go with EVD because it's less computationally complex than SVD. Reach for SVD whenever there is need to handle non-square matrices.


15. Anything that surprised you or felt counterintuitive during this lab
    (non-monotonic EVD error, complex eigenvalues, the ringing artifacts,
    anything else)?

Complex eigenvalues really took it's time to land, We pondered a lot just to end up getting the justification from the quadratic roots formula.
