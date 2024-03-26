"""
From Semantic Abstraction:

To reduce noise from irrelevant regions, we use hor
izontal flips and RGB augmentations of I and obtain a
relevancy map for each augmentation. To
reliably detect small objects, we propose a multi-scale
relevancy extractor that densely computes relevancies
at different scales and locations in a sliding window
fashion (an analogous relevancy-based approach to Li
et al.'s local attention-pooling). The final relevancy
map is averaged across all augmentations and scales.

"""

import numpy as np
from PIL import Image
from CLIP.clip import ClipWrapper, saliency_configs, imagenet_templates
from time import time
import matplotlib.pyplot as plt

# Image directory
img_path = "results/clip_exp/0/0.png"
img = np.array(Image.open(img_path).convert("RGB"))
print("Image shape:", img.shape)
h, w, c = img.shape
labels = ["window", "chair"]
prompts = ["a bad photo of {}"]

start = time()
grads = ClipWrapper.get_clip_saliency(
    img=img,
    text_labels=np.array(labels),
    prompts=prompts,
    **saliency_configs["ours"](h),
)[0]
print(f"get gradcam took {float(time() - start)} seconds", grads.shape)

# save
grads -= grads.mean(axis=0)
grads = grads.cpu().numpy()
fig, axes = plt.subplots(1, len(labels), figsize=(6 * len(labels), 5))
axes = axes.flatten()
vmin = 0.002
cmap = plt.get_cmap("jet")
vmax = 0.008
for ax, label_grad, label in zip(axes, grads, labels):
    ax.axis("off")
    ax.imshow(img)
    ax.set_title(label, fontsize=12)
    grad = np.clip((label_grad - vmin) / (vmax - vmin), a_min=0.0, a_max=1.0)
    colored_grad = cmap(grad)
    grad = 1 - grad
    colored_grad[..., -1] = grad * 0.7
    ax.imshow(colored_grad)
plt.tight_layout(pad=0)
plt.savefig("grads.png")
print("dumped relevancy to grads.png")
# plt.show()
