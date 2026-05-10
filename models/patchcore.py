import torch
import torch.nn.functional as F

class PatchCore:
    def __init__(self, model, k=1000, device="cpu"):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.k = k  # coreset size

        self.memory_bank = None
        self.features = []

        # hook 등록
        self.model.layer2.register_forward_hook(self._hook)
        self.model.layer3.register_forward_hook(self._hook)

    def _hook(self, module, input, output):
        self.features.append(output)

    def _ensure_3_channel(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(0)
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        return x

    def _extract_features(self, x):
        self.features = []
        x = self._ensure_3_channel(x)

        with torch.no_grad():
            _ = self.model(x)

        f2, f3 = self.features

        # resize
        f3 = F.interpolate(f3, size=f2.shape[-2:], mode='bilinear', align_corners=False)

        # concat
        f = torch.cat([f2, f3], dim=1)

        # patch
        B, C, H, W = f.shape
        patch = f.permute(0, 2, 3, 1).reshape(B, H*W, C)

        return patch, H, W

    def fit(self, dataset):
        patches = []

        for i in range(len(dataset)):
            img, _ = dataset[i]
            img = img.unsqueeze(0).to(self.device)

            patch, _, _ = self._extract_features(img)
            patches.append(patch)

        # concat
        patches = torch.cat(patches, dim=0)
        patches = patches.reshape(-1, patches.shape[-1])

        # coreset
        self.memory_bank = self._greedy_coreset(patches, self.k)

    def _greedy_coreset(self, x, k):
        n = x.shape[0]
        k = min(k, n)

        idx = torch.randint(0, n, (1,), device=x.device)
        selected = [idx.item()]

        dist = torch.cdist(x[idx], x).squeeze(0)

        for _ in range(k - 1):
            idx = torch.argmax(dist)
            selected.append(idx.item())

            new_dist = torch.cdist(x[idx].unsqueeze(0), x).squeeze(0)
            dist = torch.minimum(dist, new_dist)

        return x[selected]

    def predict(self, img):
        """
        1.cdist로 모든 patch와 정상 memory_bank 거리계산
        2.min_dist로 최소 거리
        patch = [p1, p2, p3]
        memory = [m1, m2]
        cdist = [[1, 2], [0, 1], [3, 2]]
        cdist = p1 vs (m1, m2), p2 vs (m1, m2), p3 vs (m1, m2)
        dist.min(dim = 1) -> 각 열에서 최소값만 -> [[1],[0],[2]]
        최솟값을 뽑는 이유는 최솟값이 클수록 더 확실한 이상이기때문에.
        최솟값들중 가장 이상한 patch 찾기위해 max로 한번더.
        """
        img = self._ensure_3_channel(img).to(self.device)

        patch, H, W = self._extract_features(img)
        patch = patch.squeeze(0)

        # 거리 계산
        dist = torch.cdist(patch, self.memory_bank)
        #memory_bank와 patch간 거리들을 각각 위치별로 모두 계산후 반환.

        min_dist, _ = dist.min(dim=1)

        # image score
        score = min_dist.max()

        # heatmap
        heatmap = min_dist.reshape(H, W)

        return score, heatmap

    def predict_batch(self, imgs):
        scores = []
        heatmaps = []
        for img in imgs:
            score, heatmap = self.predict(img)
            scores.append(score)
            heatmaps.append(heatmap)
        return torch.stack(scores), heatmaps

