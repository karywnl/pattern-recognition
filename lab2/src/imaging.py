import matplotlib.image as mpimg
import numpy as np
from PIL import Image

# load a PNG as an (H, W, 3) uint8 RGB array
def load_image(path):
    img = mpimg.imread(path)
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)
    return img[..., :3]

# resize to (new_width, new_height)
def resize(img, size):
    pil_img = Image.fromarray(img)
    resized = pil_img.resize(size)
    return np.array(resized)

# resize so the longer side becomes max_side, keeping aspect ratio
def resize_preserve_aspect(img, max_side):
    h, w = img.shape[:2]
    if w >= h:
        new_w = max_side
        new_h = round(h * max_side / w)
    else:
        new_h = max_side
        new_w = round(w * max_side / h)
    return resize(img, (new_w, new_h))

# convert (H, W, 3) RGB to (H, W) grayscale
def to_grayscale(img):
    R = img[..., 0].astype(float)
    G = img[..., 1].astype(float)
    B = img[..., 2].astype(float)
    return 0.299 * R + 0.587 * G + 0.114 * B
