import numpy as np
from PIL import Image
from scipy.ndimage import zoom
from scipy.interpolate import griddata
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
from collections import OrderedDict
from io import BytesIO

def unwrap2D(wrapped_image, quality_image=None) -> np.ndarray:
    
    class Pixel:
        def __init__(self, value: float, reliability: float):
            self.value = value
            self.reliability = reliability
            self.increment = 0
            self.size = 1
            self.head = self
            self.last = self
            self.next = None

    class Edge:
        def __init__(self, p1: Pixel, p2: Pixel):
            self.p1 = p1
            self.p2 = p2
            self.reliability = p1.reliability + p2.reliability
            diff = p1.value - p2.value
            self.increment = -1 if diff > 0.5 else (1 if diff < -0.5 else 0)

    def wrap(pixel_value):
        return pixel_value +  np.where(pixel_value > 0.5, -1, np.where(pixel_value < -0.5, 1, 0))
    
    def miguel_quality(wrapped):
            
        WIP = wrapped[1:-1, 1:-1]
        
        H  = wrap(wrapped[1:-1, :-2] - WIP) - wrap(WIP - wrapped[1:-1, 2:])
        V  = wrap(wrapped[:-2, 1:-1] - WIP) - wrap(WIP - wrapped[2:, 1:-1])
        D1 = wrap(wrapped[:-2, :-2] - WIP)  - wrap(WIP - wrapped[2:, 2:])
        D2 = wrap(wrapped[:-2, 2:] - WIP)   - wrap(WIP - wrapped[2:, :-2])
    
        reliability = 1.0/(H**2 + V**2 + D1**2 + D2**2)
    
        miguel_array=np.zeros_like(wrapped)
        miguel_array[1:-1, 1:-1] = np.sqrt(reliability)
        return miguel_array

    if quality_image is None:
        quality_image = miguel_quality(wrapped_image)
        
    if wrapped_image.shape != quality_image.shape:
        raise ValueError("Input arrays must have the same dimensions")
    
    height, width = wrapped_image.shape
    
    # Flatten images into list of Pixel objects
    pixels = np.array([
        Pixel(wrapped_image.flat[i], quality_image.flat[i]) for i in range(wrapped_image.size)
    ])

    # Build edges between horizontal and vertical neighbors
    edges = []
    for r in range(height):
        for c in range(width):
            idx = r * width + c
            if c < width - 1:
                edges.append(Edge(pixels[idx], pixels[idx + 1]))
            if r < height - 1:
                edges.append(Edge(pixels[idx], pixels[idx + width]))

    # Sort edges by reliability in descending order
    edges.sort(key=lambda e: e.reliability, reverse=True)

    def merge_groups(p1: Pixel, p2: Pixel, edge_increment: int):
        group1 = p1.head
        group2 = p2.head

        if group1.size < group2.size:
            group1, group2 = group2, group1
            p1, p2 = p2, p1
            edge_increment = -edge_increment

        # Merge group2 into group1
        group1.last.next = group2
        group1.last = group2.last
        group1.size += group2.size

        delta = p1.increment - edge_increment - p2.increment
        while group2:
            group2.head = group1
            group2.increment += delta
            group2 = group2.next

    # Process edges to unwrap phase
    for edge in edges:
        if edge.p1.head != edge.p2.head:
            merge_groups(edge.p1, edge.p2, edge.increment)

    unwrapped = np.array([p.value + p.increment for p in pixels]).reshape((height, width))
    return unwrapped
def plot(images_dict):
    n = len(images_dict)
    ncols = int(np.ceil(np.sqrt(n)))
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(8, 8))
    axes = axes.flatten()

    for i, (key, image) in enumerate(images_dict.items()):
        im = axes[i].imshow(image)
        fig.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
        axes[i].set_title(key)
        axes[i].grid()

    for j in range(n, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    fig.savefig("/output.png")  # Save to Pyodide virtual FS

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

def filterAngleInterfringe(fft, thick, anglerad, interfringe):
    dy, dx = fft.shape
    sr = np.sin(anglerad)
    cr = np.cos(anglerad)
    freq_filter = interfringe / np.sqrt((cr*dx)**2 + (sr*dy)**2)
    slit_freq = (np.pi * thick) / np.sqrt((sr*dx)**2 + (cr*dy)**2)
    X, Y = np.meshgrid(np.arange(-dx//2, dx-dx//2), np.arange(-dy//2, dy-dy//2))
    Xr = cr*X + sr*Y
    Yr = -sr*X + cr*Y
    morlet = np.exp(-(np.pi*(Xr*freq_filter - 1))**2 - (Yr*slit_freq)**2) * np.sqrt(np.sqrt(thick))
    filtered = np.fft.fftshift(morlet) * fft
    signal = np.fft.ifft2(filtered)
    contrast = 2 * np.abs(signal)
    fringeshift = np.arctan2(np.imag(signal), np.real(signal)) / (2 * np.pi)
    return fringeshift, contrast

def interpolate(fringeshift, bad_mask):
    height, width = fringeshift.shape
    x = np.linspace(0, 10, width)
    y = np.linspace(0, 10, height)
    X, Y = np.meshgrid(x, y)
    points = np.array((Y[~bad_mask], X[~bad_mask])).T
    values = fringeshift[~bad_mask]
    interp_points = np.array((Y[bad_mask], X[bad_mask])).T
    interpolated_values = griddata(points, values, interp_points, method='linear')
    data_interp = fringeshift.copy()
    data_interp[bad_mask] = interpolated_values
    data_interp -= np.nanmin(fringeshift)
    data_interp = np.nan_to_num(data_interp, nan=0.0)
    return data_interp

def analyze(FileRef, FileShot, scale=None, weight=0.5, wl=[1,1,1], al=[1,1,1], tl=[1,1,1], cutoff=0, invertSign=False):
    ref = np.array(Image.open(FileRef))
    shot = np.array(Image.open(FileShot))   
    images_dict = OrderedDict()

    if scale:
        zoom_factors = np.array(scale) / np.array(ref.shape)
        ref = zoom(ref, zoom_factors)
        shot = zoom(shot, zoom_factors)

    fftRef = np.fft.fft2(ref)
    fftShot = np.fft.fft2(shot)

    anglerad, interfringe = guess(fftRef, weight)

    interfringes = interfringe * np.linspace(*wl if wl[2] > 0 else [1, 1, 1])
    thicknesses = interfringe * np.linspace(*tl if tl[2] > 0 else [1, 1, 1])
    angles = anglerad + (np.deg2rad(np.linspace(*al)) if al[2] > 0 else [0, 0, 1])

    fringeshiftRef, contrastRef = filterAngleInterfringe(fftRef, interfringe, anglerad, interfringe)

    bestContrast = np.zeros_like(contrastRef)
    bestFringeshift = np.zeros_like(contrastRef)
    bestInterfringe = np.zeros_like(contrastRef)
    bestAngle = np.zeros_like(contrastRef)
    bestThick = np.zeros_like(contrastRef)

    for t in thicknesses:
        for i in interfringes:
            for a in angles:
                fringeshift, contrast = filterAngleInterfringe(fftShot, t, a, i)
                bestC = contrast > bestContrast
                bestContrast[bestC] = contrast[bestC]
                bestFringeshift[bestC] = fringeshift[bestC]
                bestThick[bestC] = t
                bestInterfringe[bestC] = i
                bestAngle[bestC] = a / np.pi

    unwrapRef = unwrap2D(fringeshiftRef, contrastRef)
    unwrapAngles = unwrap2D(bestAngle, bestContrast)
    swaps = np.rint(bestAngle - unwrapAngles).astype(int) % 2
    bestFringeshift[swaps == 1] = -bestFringeshift[swaps == 1]
    unwrapShot = unwrap2D(bestFringeshift, bestContrast)

    diff = lambda arr: np.max(arr) - np.min(arr)
    fringeshift = unwrapShot - unwrapRef if diff(unwrapShot - unwrapRef) < diff(unwrapShot + unwrapRef) else unwrapShot + unwrapRef
    if invertSign:
        fringeshift = -fringeshift

    cutoff_value = np.min(bestContrast) + cutoff * (np.max(bestContrast) - np.min(bestContrast))
    cutoff_mask = bestContrast < cutoff_value
    fringeshift[cutoff_mask] = np.nan
    fringeshift -= np.nanmin(fringeshift)
    interpolated = interpolate(fringeshift, cutoff_mask)

    bestThick[cutoff_mask] = np.nan
    bestInterfringe[cutoff_mask] = np.nan
    unwrapAngles[cutoff_mask] = np.nan

    images_dict['synthetic'] = bestContrast * (1 + np.cos(bestFringeshift * 2 * np.pi))
    images_dict['contrast'] = bestContrast
    images_dict['fringeshift'] = fringeshift
    images_dict['interpolated'] = interpolated
#     images_dict['swaps'] = swaps
#     images_dict['angle'] = bestAngle - anglerad / np.pi
#     images_dict['interfringe'] = bestInterfringe / interfringe

    image = Image.fromarray(interpolated)
    image.save("/output.tiff", format='TIFF')
    
    return images_dict

