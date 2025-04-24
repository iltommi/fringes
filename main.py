import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
from collections import OrderedDict
from io import BytesIO
import sys

def unwrap2D(wrapped_image, quality_image=None) -> np.ndarray:
    def wrap(x): return x + np.where(x > 0.5, -1, np.where(x < -0.5, 1, 0))

    def compute_quality(wrapped):
        W = wrapped[1:-1, 1:-1]
        diffs = [
            wrap(wrapped[1:-1, :-2] - W) - wrap(W - wrapped[1:-1, 2:]),
            wrap(wrapped[:-2, 1:-1] - W) - wrap(W - wrapped[2:, 1:-1]),
            wrap(wrapped[:-2, :-2] - W) - wrap(W - wrapped[2:, 2:]),
            wrap(wrapped[:-2, 2:] - W) - wrap(W - wrapped[2:, :-2])
        ]
        reliability = 1.0 / sum(d**2 for d in diffs)
        result = np.zeros_like(wrapped)
        result[1:-1, 1:-1] = np.sqrt(reliability)
        return result

    if quality_image is None:
        quality_image = compute_quality(wrapped_image)
    if wrapped_image.shape != quality_image.shape:
        raise ValueError("Input arrays must have the same dimensions")

    h, w = wrapped_image.shape
    size = wrapped_image.size

    class P:
        def __init__(self, v, r):
            self.v, self.r, self.i, self.s, self.h, self.l, self.n = v, r, 0, 1, self, self, None

    pixels = np.array([P(wrapped_image.flat[i], quality_image.flat[i]) for i in range(size)])

    class E:
        def __init__(self, p1, p2):
            d = p1.v - p2.v
            self.p1, self.p2 = p1, p2
            self.r = p1.r + p2.r
            self.i = -1 if d > 0.5 else (1 if d < -0.5 else 0)

    edges = [E(pixels[r*w+c], pixels[r*w+c+1]) for r in range(h) for c in range(w-1)] + \
            [E(pixels[r*w+c], pixels[(r+1)*w+c]) for r in range(h-1) for c in range(w)]
    edges.sort(key=lambda e: e.r, reverse=True)

    def merge(p1, p2, inc):
        g1, g2 = p1.h, p2.h
        if g1.s < g2.s: g1, g2, p1, p2, inc = g2, g1, p2, p1, -inc
        g1.l.n, g1.l, g1.s = g2, g2.l, g1.s + g2.s
        delta = p1.i - inc - p2.i
        while g2: g2.h, g2.i, g2 = g1, g2.i + delta, g2.n

    for e in edges:
        if e.p1.h != e.p2.h:
            merge(e.p1, e.p2, e.i)

    return np.array([p.v + p.i for p in pixels]).reshape((h, w))

def scale_array(arr, new_shape):
    old_rows, old_cols = arr.shape
    new_rows, new_cols = new_shape
    
    old_row_indices = np.linspace(0, old_rows - 1, new_rows)
    old_col_indices = np.linspace(0, old_cols - 1, new_cols)
    
    old_row_grid, old_col_grid = np.meshgrid(old_row_indices, old_col_indices, indexing='ij')
    
    scaled_arr = arr[old_row_grid.astype(int), old_col_grid.astype(int)]
    
    return scaled_arr
    
def guess(fftRef, weight):
    dy, dx = fftRef.shape
    fx = np.fft.fftfreq(dx)
    fy = np.fft.fftfreq(dy)
    X, Y = np.meshgrid(fx, fy)
    mag = np.abs(fftRef) * (X**2 + Y**2)**0.5
    mymax = np.unravel_index(mag.argmax(), mag.shape)
    fxm, fym = fx[mymax[1]], fy[mymax[0]]
    anglerad = np.arctan2(fym, fxm)
    interfringe = 1 / np.sqrt(fxm**2 + fym**2)
    return anglerad, interfringe

def filterAngleInterfringe(fft, anglerad, interfringe):
    dy, dx = fft.shape
    sr = np.sin(anglerad)
    cr = np.cos(anglerad)
    freq_filter = interfringe / np.sqrt((cr*dx)**2 + (sr*dy)**2)
    slit_freq = (np.pi * interfringe) / np.sqrt((sr*dx)**2 + (cr*dy)**2)
    X, Y = np.meshgrid(np.arange(-dx//2, dx-dx//2), np.arange(-dy//2, dy-dy//2))
    Xr = cr*X + sr*Y
    Yr = -sr*X + cr*Y
    morlet = np.exp(-(np.pi*(Xr*freq_filter - 1))**2 - (Yr*slit_freq)**2) 
    filtered = np.fft.fftshift(morlet) * fft
    signal = np.fft.ifft2(filtered)
    contrast = 2 * np.abs(signal)
    fringeshift = np.arctan2(np.imag(signal), np.real(signal)) / (2 * np.pi)
    return fringeshift, contrast


def analyze(FileRef, FileShot, wl=1, al=1, cutoff=0):
    
    ref = np.array(Image.open(FileRef).convert('L'))
    shot = np.array(Image.open(FileShot).convert('L'))   
    images_dict = OrderedDict()

    orig_size=shot.shape
    scale=256    
    ref = scale_array(ref, (scale,scale))
    shot = scale_array(shot, (scale,scale))

    fftRef = np.fft.fft2(ref)
    fftShot = np.fft.fft2(shot)

    weight=0.5
    anglerad, interfringe = guess(fftRef, weight)

    i0=3
    wl = wl if wl%2==1 else wl+1
    interfringes = [i0 * (((interfringe / i0) ** (1 / (wl // 2))) ** i) for i in range(wl)]
    al=al if al%2==1 else al+1
    angles      = anglerad+(np.deg2rad(np.arange(-90,90,180/al)))

    fringeshiftRef, contrastRef = filterAngleInterfringe(fftRef, anglerad, interfringe)

    bestContrast = np.zeros_like(contrastRef)
    bestFringeshift = np.zeros_like(contrastRef)
    bestInterfringe = np.zeros_like(contrastRef)
    bestAngle = np.zeros_like(contrastRef)

    for i in interfringes:
        for a in angles:
            fringeshift, contrast = filterAngleInterfringe(fftShot, a, i)
            bestC = contrast > bestContrast
            bestContrast[bestC] = contrast[bestC]
            bestFringeshift[bestC] = fringeshift[bestC]
            bestInterfringe[bestC] = i
            bestAngle[bestC] = a / np.pi

    unwrapRef = unwrap2D(fringeshiftRef, contrastRef)
    unwrapAngles = unwrap2D(bestAngle, bestContrast)
    swaps = np.rint(bestAngle - unwrapAngles).astype(int) % 2
    bestFringeshift[swaps == 1] = -bestFringeshift[swaps == 1]
    unwrapShot = unwrap2D(bestFringeshift, bestContrast)

    diff = lambda arr: np.max(arr) - np.min(arr)
    fringeshift = unwrapShot - unwrapRef if diff(unwrapShot - unwrapRef) < diff(unwrapShot + unwrapRef) else unwrapShot + unwrapRef

    cutoff_value = np.min(bestContrast) + cutoff * (np.max(bestContrast) - np.min(bestContrast))
    cutoff_mask = bestContrast < cutoff_value
    fringeshift[cutoff_mask] = np.nan
    fringeshift -= np.nanmin(fringeshift)

    bestInterfringe[cutoff_mask] = np.nan
    unwrapAngles[cutoff_mask] = np.nan

    images_dict['Original'] = shot
    images_dict['Synthetic'] = bestContrast * (1 + np.cos(bestFringeshift * 2 * np.pi))
    images_dict['Fringeshift'] = fringeshift

    fig, axes = plt.subplots(1, len(images_dict), figsize=(10, 3))
    axes = axes.flatten()

    for i, (key, image) in enumerate(images_dict.items()):
        image = scale_array(image,orig_size)
        im = axes[i].imshow(image)
        fig.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
        axes[i].set_title(key)
        axes[i].grid()

    plt.tight_layout()
    directory= '/' if sys.platform == "emscripten" else ""
    fig.savefig(directory+"output.png") 
    image = Image.fromarray(fringeshift)
    image.save(directory+"output.tiff", format='TIFF')
        