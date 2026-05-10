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

def show_result(dataset, scores, heatmaps, k=4):
    topk_idx = np.argsort(scores)[-k:][::-1]

    plt.figure(figsize=(3*k, 6))

    for i, idx in enumerate(topk_idx):
        img, label = dataset[idx]
        image = denormalize(img).permute(1, 2, 0).numpy()
        label_name = dataset.classes[label]
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
    plt.show()
