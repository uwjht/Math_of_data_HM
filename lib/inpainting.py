import pywt
import numpy as np
from lib.opt_types import *
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
np.random.seed(0)

m = 256
one_coeffs = pywt.coeffs_to_array(pywt.wavedec2(np.ones((m, m)), 'db8', mode='periodization'))[1]

mask = []
for i in range(m):
    for j in range(m):
        mask.append(np.random.binomial(1, 0.4))
mask = np.array(mask)

def P(x: Vector) -> Vector:
    return x[mask==1]

def P_T(x: Vector) -> Vector:
    u = np.zeros(m*m)
    u[mask==1] = x
    return u

def W(x: Vector) -> Vector:
    x = x.reshape(m, m)
    coeffs = pywt.wavedec2(x, 'db8', mode='periodization') 
    wav, _ = pywt.coeffs_to_array(coeffs)
    return wav.reshape(-1)

def W_T(x: Vector) -> Vector:
    wav_x = x.reshape(m, m)
    u = pywt.array_to_coeffs(wav_x, one_coeffs, output_format='wavedec2')
    x = pywt.waverec2(u, 'db8', mode='periodization')
    return x.reshape(-1)


def show_subsampled(image):
    sub = (mask*image.reshape(-1)).reshape(m, m)
    plt.imshow(sub, cmap='gray')
    return sub


def load(image_path):
    im = Image.open(image_path)
    im = im.resize((256, 256))
    im = ImageOps.grayscale(im)

    im = np.array(im)
    plt.imshow(im, cmap='gray')
    return im

def solve_composite(method: OptAlgorithm, composite_objective: CompositeFunction, lmda: float, max_iterations: int) -> Vector:
    x_lst = []
    f_val_lst = []
    g_val_lst = []
    alpha_zero = np.zeros(m*m)
    composite_objective.g.lmda = lmda
    state = method.init_state(composite_objective, alpha_zero)
    for _ in range(max_iterations):
        state = method.state_update(composite_objective, state)
        x_lst.append(state.x_k)
        f_val_lst.append(composite_objective.f(state.x_k))  
        g_val_lst.append(composite_objective.g(state.x_k))  
        
        
    f, (ax_1, ax_2, ax_3) = plt.subplots(1, 3, figsize=(24, 4))
    # f.suptitle("TEST OF  " + method.name + " ", fontsize=14)

    ax_1.plot(
            range(max_iterations),
            np.array(f_val_lst),
            color=(0, 0, 1),
            lw=2,
            label=method.name,
        )
    ax_1.legend(fontsize=14)
    ax_1.set_xlabel("#iterations", fontsize=14)
    ax_1.set_ylabel(r"$f(\mathbf{x}^k)$", fontsize=14)
    ax_1.set_yscale("log")
    ax_1.grid()
    
    ax_2.plot(
            range(max_iterations),
            np.array(g_val_lst),
            color=(0, 0, 1),
            lw=2,
            label=method.name,
        )
    ax_2.legend(fontsize=14)
    ax_2.set_xlabel("#iterations", fontsize=14)
    ax_2.set_ylabel(r"$g(\mathbf{x}^k)$", fontsize=14)
    ax_2.set_yscale("log")
    ax_2.grid()
    
    ax_3.plot(
            range(max_iterations),
            np.array(f_val_lst) + np.array(g_val_lst),
            color=(0, 0, 1),
            lw=2,
            label=method.name,
        )
    ax_3.legend(fontsize=14)
    ax_3.set_xlabel("#iterations", fontsize=14)
    ax_3.set_ylabel(r"$f(\mathbf{x}^k) + g(\mathbf{x}^k)$", fontsize=14)
    ax_3.set_yscale("log")
    ax_3.grid()

    return state.x_k

def show(true, subsampled, estimated):
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))

    ax1.imshow(true, cmap='gray')
    ax1.set_title("Original Image")
    ax1.set_xticks([])
    ax1.set_yticks([])

    ax2.imshow(subsampled, cmap='gray')
    ax2.set_title("Subsampled Image")
    ax2.set_xticks([])
    ax2.set_yticks([])

    ax3.imshow(estimated, cmap='gray')
    ax3.set_title("Reconstructed Image")
    ax3.set_xticks([])
    ax3.set_yticks([])

    plt.show()
