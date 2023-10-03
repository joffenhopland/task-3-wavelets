import numpy as np
import matplotlib.pyplot as plt
import pywt
import pywt.data


def plot_coeffs(data, w, titles, use_color=True):
    # plot the wavelet coefficients
    fig, axes = plt.subplots(1, len(data), figsize=(12, 3))
    max_ = max(tuple(np.max(np.abs(v)) for v in data))

    if use_color:
        if w == "db1":
            cmap = plt.cm.bwr
        else:
            cmap = plt.cm.seismic

    for ax, coeffs, title in zip(axes, data, titles):
        if use_color:
            ax.imshow(coeffs, interpolation="nearest", cmap=cmap, vmin=-max_, vmax=max_)
        else:
            ax.imshow(coeffs, interpolation="nearest", cmap=plt.cm.gray)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    plt.show()


# load image
image = pywt.data.camera()

# 2D wavelet decomposition
coeffs = pywt.wavedec2(image, "db1", level=2)
cA2, (cH2, cV2, cD2), (cH1, cV1, cD1) = coeffs

# original image
plt.figure()
plt.imshow(image, cmap="gray")
plt.title("Original Image")
plt.axis("off")
plt.show()

# approximation and details
titles = ["Approximation", " Horizontal detail", "Vertical detail", "Diagonal detail"]
plot_coeffs([cA2, cH2, cV2, cD2], "db1", titles)

# reconstruct image
reconstructed = pywt.waverec2(coeffs, "db1")
plt.figure()
plt.imshow(reconstructed, cmap="gray")
plt.title("Reconstructed Image")
plt.axis("off")
plt.show()
