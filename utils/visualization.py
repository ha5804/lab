import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import numpy as np

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

def denormalize(img):
    """
    정규화된 이미지를 원래 픽셀 값으로 복원
    args:
        img(Tensor) : (C, H, W)
    returns:
        Tensor: (C, H, W) 형태 복원 이미지 (0~1)
    """
    img = img.detach().cpu() * IMAGENET_STD + IMAGENET_MEAN
    return img.clamp(0, 1)

def show_result(dataset, scores, heatmaps, k=4, save_path=None, show=True):
    topk_idx = np.argsort(scores)[-k:][::-1]

    fig = plt.figure(figsize=(3*k, 6))

    for i, idx in enumerate(topk_idx):
        img, label = dataset[idx]
        image = denormalize(img).permute(1, 2, 0).numpy()
        label_name = dataset.classes[int(label)]
        heatmap = heatmaps[idx].detach().cpu()
        heatmap = F.interpolate(
            heatmap.unsqueeze(0).unsqueeze(0),
            size=image.shape[:2],
            mode="bilinear",
            align_corners=False,
        ).squeeze()

        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        heatmap = heatmap.numpy()

        plt.subplot(2, k, i+1)
        plt.imshow(image)
        plt.title(f"{scores[idx]:.2f}\n{label_name}")
        plt.axis("off")

        plt.subplot(2, k, k+i+1)
        plt.imshow(heatmap, cmap="jet")
        plt.axis("off")

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def save_heatmap(image, heatmap, save_path, score=None, label_name=None):
    image = denormalize(image).permute(1, 2, 0).numpy()
    heatmap = heatmap.detach().cpu()
    heatmap = F.interpolate(
        heatmap.unsqueeze(0).unsqueeze(0),
        size=image.shape[:2],
        mode="bilinear",
        align_corners=False,
    ).squeeze()
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    heatmap = heatmap.numpy()

    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    title = []
    if score is not None:
        title.append(f"{float(score):.4f}")
    if label_name is not None:
        title.append(str(label_name))

    axes[0].imshow(image)
    axes[0].set_title("image")
    axes[0].axis("off")

    axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title("heatmap")
    axes[1].axis("off")

    axes[2].imshow(image)
    axes[2].imshow(heatmap, cmap="jet", alpha=0.45)
    axes[2].set_title("overlay")
    axes[2].axis("off")

    if title:
        fig.suptitle(" / ".join(title))
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
