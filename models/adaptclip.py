import torch
import open_clip
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import functional as TF


valid_backbones = ["ViT-B-16-plus-240", "ViT-L-14-336"]
valid_pretrained_datasets = ["laion400m_e32", "openai"]
ADAPTCLIP_HF_REPO = "csgaobb/AdaptCLIP"
ADAPTCLIP_PRETRAINED_CHECKPOINTS = {
    "mvtec": "adaptclip_checkpoints/12_4_128_train_on_mvtec_3adapters_batch8/epoch_15.pth",
    "visa": "adaptclip_checkpoints/12_4_128_train_on_visa_3adapters_batch8/epoch_15.pth",
}

mean_train = [0.48145466, 0.4578275, 0.40821073]
std_train = [0.26862954, 0.26130258, 0.27577711]


def _convert_to_rgb(image):
    return image.convert("RGB")


class ResidualMLP(nn.Module):
    def __init__(self, dim, hidden_dim=None, dropout=0.0):
        super().__init__()
        hidden_dim = hidden_dim or max(dim // 4, 64)
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )
        nn.init.zeros_(self.net[1].bias)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return x + self.net(x)


class PromptQueryHead(nn.Module):
    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or max(dim // 4, 64)
        self.local = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.global_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, joint_tokens):
        local_score = torch.sigmoid(self.local(joint_tokens)).squeeze(-1)
        avg_pool = joint_tokens.mean(dim=1)
        max_pool = joint_tokens.max(dim=1)[0]
        global_score = torch.sigmoid(self.global_head((avg_pool + max_pool) * 0.5)).squeeze(-1)
        return local_score, global_score


class OpenClipAdaptVisual(nn.Module):
    def __init__(self, visual):
        super().__init__()
        self.visual = visual
        self.grid_size = self._get_grid_size()

    def _get_grid_size(self):
        grid_size = getattr(self.visual, "grid_size", None)
        if grid_size is not None:
            if isinstance(grid_size, int):
                return (grid_size, grid_size)
            return tuple(grid_size)

        image_size = getattr(self.visual, "image_size", 240)
        patch_size = getattr(self.visual, "patch_size", 16)
        image_h, image_w = image_size if isinstance(image_size, tuple) else (image_size, image_size)
        patch_h, patch_w = patch_size if isinstance(patch_size, tuple) else (patch_size, patch_size)
        return (image_h // patch_h, image_w // patch_w)

    def forward(self, image):
        visual = self.visual
        if not all(hasattr(visual, name) for name in ["conv1", "class_embedding", "positional_embedding", "ln_pre", "transformer", "ln_post"]):
            raise NotImplementedError("AdaptCLIP currently needs an open_clip VisionTransformer visual backbone.")

        x = visual.conv1(image)
        grid_h, grid_w = x.shape[-2:]
        self.grid_size = (grid_h, grid_w)

        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        cls_token = visual.class_embedding.to(x.dtype)
        cls_token = cls_token + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self._positional_embedding(x, grid_h, grid_w)

        patch_dropout = getattr(visual, "patch_dropout", None)
        if patch_dropout is not None:
            x = patch_dropout(x)

        x = visual.ln_pre(x)
        x = x.permute(1, 0, 2)
        x = visual.transformer(x)
        x = x.permute(1, 0, 2)
        x = visual.ln_post(x)

        proj = getattr(visual, "proj", None)
        if proj is not None:
            x = x @ proj

        global_token = x[:, 0]
        patch_tokens = x[:, 1:]
        return global_token, patch_tokens

    def _positional_embedding(self, x, grid_h, grid_w):
        pos = self.visual.positional_embedding.to(dtype=x.dtype, device=x.device)
        if pos.shape[0] == x.shape[1]:
            return pos

        class_pos = pos[:1]
        patch_pos = pos[1:]
        old_grid = int(patch_pos.shape[0] ** 0.5)
        patch_pos = patch_pos.reshape(1, old_grid, old_grid, -1).permute(0, 3, 1, 2)
        patch_pos = F.interpolate(patch_pos, size=(grid_h, grid_w), mode="bicubic", align_corners=False)
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(grid_h * grid_w, -1)
        return torch.cat([class_pos, patch_pos], dim=0)


class AdaptCLIPModel(nn.Module):
    def __init__(
        self,
        out_size_h,
        out_size_w,
        device,
        backbone="ViT-L-14-336",
        pretrained_dataset="openai",
        precision="fp32",
        checkpoint_path=None,
    ):
        super().__init__()
        if backbone not in valid_backbones:
            raise ValueError(f"Unsupported AdaptCLIP backbone: {backbone}")
        if pretrained_dataset not in valid_pretrained_datasets:
            raise ValueError(f"Unsupported AdaptCLIP pretrained dataset: {pretrained_dataset}")

        self.out_size_h = out_size_h
        self.out_size_w = out_size_w
        self.device = device
        self.precision = precision

        clip_model, _, _ = open_clip.create_model_and_transforms(
            backbone,
            pretrained=pretrained_dataset,
            precision=precision,
        )
        self.clip_model = clip_model
        self.visual = OpenClipAdaptVisual(clip_model.visual)
        self.tokenizer = open_clip.get_tokenizer(backbone)

        embed_dim = self._get_embed_dim()
        self.local_visual_adapter = ResidualMLP(embed_dim)
        self.global_visual_adapter = ResidualMLP(embed_dim)
        self.textual_adapter = nn.Parameter(torch.empty(2, embed_dim))
        self.prompt_query_head = PromptQueryHead(embed_dim)
        nn.init.zeros_(self.textual_adapter)

        self.text_features = None
        self.prompt_patch_gallery = None
        self.prompt_global_gallery = None
        self.has_trained_prompt_query = False
        self.grid_size = self.visual.grid_size

        self.to(device)
        self.clip_model.eval()
        for param in self.clip_model.parameters():
            param.requires_grad = False

        if checkpoint_path is not None:
            self.load_adapter_checkpoint(checkpoint_path)

    def _get_embed_dim(self):
        proj = getattr(self.visual.visual, "proj", None)
        if proj is not None:
            return proj.shape[-1]

        text_projection = getattr(self.clip_model, "text_projection", None)
        if text_projection is not None:
            return text_projection.shape[-1]

        output_dim = getattr(self.clip_model, "output_dim", None)
        if output_dim is not None:
            return output_dim

        raise ValueError("Could not infer CLIP embedding dimension for AdaptCLIP.")

    def load_adapter_checkpoint(self, checkpoint_path):
        checkpoint_path = self._resolve_checkpoint_path(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state_dict = self._checkpoint_state_dict(checkpoint)
        state_dict = self._strip_common_prefixes(state_dict)
        compatible_state_dict = self._compatible_state_dict(state_dict)
        _, unexpected = self.load_state_dict(compatible_state_dict, strict=False)
        self.has_trained_prompt_query = any(key.startswith("prompt_query_head.") for key in compatible_state_dict)
        print(f"Loaded AdaptCLIP checkpoint: {checkpoint_path}")
        print(f"Loaded compatible AdaptCLIP keys: {len(compatible_state_dict)} / {len(state_dict)}")
        if not compatible_state_dict:
            print("No compatible AdaptCLIP adapter keys were loaded.")
        if unexpected:
            print(f"AdaptCLIP checkpoint unexpected keys: {unexpected}")

    def _checkpoint_state_dict(self, checkpoint):
        if all(key in checkpoint for key in ["textual_learner", "visual_learner", "pq_learner"]):
            return self._convert_official_checkpoint(checkpoint)
        return checkpoint.get("state_dict", checkpoint)

    def _convert_official_checkpoint(self, checkpoint):
        converted = {}
        textual = checkpoint["textual_learner"]
        visual = checkpoint["visual_learner"]

        if "ctx_pos" in textual and "ctx_neg" in textual:
            ctx_pos = textual["ctx_pos"].reshape(-1, textual["ctx_pos"].shape[-1]).mean(dim=0)
            ctx_neg = textual["ctx_neg"].reshape(-1, textual["ctx_neg"].shape[-1]).mean(dim=0)
            converted["textual_adapter"] = torch.stack([ctx_pos, ctx_neg], dim=0)

        key_map = {
            "local_adater.fc.0.weight": "local_visual_adapter.net.1.weight",
            "local_adater.fc.3.weight": "local_visual_adapter.net.4.weight",
            "global_adapter.fc.0.weight": "global_visual_adapter.net.1.weight",
            "global_adapter.fc.3.weight": "global_visual_adapter.net.4.weight",
        }
        for source_key, target_key in key_map.items():
            if source_key in visual:
                converted[target_key] = visual[source_key]

        return converted

    def _resolve_checkpoint_path(self, checkpoint_path):
        if checkpoint_path in ADAPTCLIP_PRETRAINED_CHECKPOINTS:
            try:
                from huggingface_hub import hf_hub_download
            except ImportError as exc:
                raise ImportError("Install huggingface_hub or pass a local checkpoint path.") from exc
            return hf_hub_download(
                repo_id=ADAPTCLIP_HF_REPO,
                filename=ADAPTCLIP_PRETRAINED_CHECKPOINTS[checkpoint_path],
            )
        return checkpoint_path

    def _strip_common_prefixes(self, state_dict):
        cleaned = {}
        prefixes = ("module.", "model.", "adaptclip.", "net.")
        for key, value in state_dict.items():
            clean_key = key
            changed = True
            while changed:
                changed = False
                for prefix in prefixes:
                    if clean_key.startswith(prefix):
                        clean_key = clean_key[len(prefix):]
                        changed = True
            cleaned[clean_key] = value
        return cleaned

    def _compatible_state_dict(self, state_dict):
        current_state = self.state_dict()
        compatible = {}
        skipped = []
        for key, value in state_dict.items():
            if key not in current_state:
                skipped.append(key)
                continue
            if current_state[key].shape != value.shape:
                skipped.append(key)
                continue
            compatible[key] = value
        if skipped:
            print(f"Skipped incompatible AdaptCLIP keys: {len(skipped)}")
        return compatible

    @torch.no_grad()
    def encode_image(self, images):
        if self.precision == "fp16":
            images = images.half()
        global_token, patch_tokens = self.visual(images)
        self.grid_size = self.visual.grid_size
        global_token = F.normalize(global_token.float(), dim=-1)
        patch_tokens = F.normalize(patch_tokens.float(), dim=-1)
        return global_token, patch_tokens

    @torch.no_grad()
    def encode_text_prompts(self, category):
        normal_prompts = [
            f"a photo of a flawless {category}",
            f"a photo of a perfect {category}",
            f"a photo of a normal {category}",
            f"a close-up photo of an unbroken {category}",
        ]
        abnormal_prompts = [
            f"a photo of a broken {category}",
            f"a photo of a damaged {category}",
            f"a photo of a defective {category}",
            f"a close-up photo of a corrupted {category}",
        ]
        normal_tokens = self.tokenizer(normal_prompts).to(self.device)
        abnormal_tokens = self.tokenizer(abnormal_prompts).to(self.device)
        normal_features = self.clip_model.encode_text(normal_tokens).float()
        abnormal_features = self.clip_model.encode_text(abnormal_tokens).float()
        normal_feature = normal_features.mean(dim=0, keepdim=True)
        abnormal_feature = abnormal_features.mean(dim=0, keepdim=True)
        text_features = torch.cat([normal_feature, abnormal_feature], dim=0)
        return F.normalize(text_features, dim=-1)

    def build_text_feature_gallery(self, category):
        static_text_features = self.encode_text_prompts(category)
        adapted_text_features = static_text_features + self.textual_adapter
        self.text_features = F.normalize(adapted_text_features, dim=-1)

    def text_anomaly_map(self, patch_tokens, use_visual_adapter=False):
        if use_visual_adapter:
            patch_tokens = F.normalize(self.local_visual_adapter(patch_tokens), dim=-1)
        logits = 100.0 * patch_tokens @ self.text_features.T
        return logits.softmax(dim=-1)[..., 1]

    def prompt_query_map(self, patch_tokens):
        if self.prompt_patch_gallery is None:
            return None, None

        gallery = self.prompt_patch_gallery.to(patch_tokens.device)
        similarity = patch_tokens @ gallery.T
        nearest_index = similarity.argmax(dim=-1)
        aligned_prompt = gallery[nearest_index]
        residual = (patch_tokens - aligned_prompt).abs()
        joint_tokens = patch_tokens + residual

        distance_map = 1.0 - similarity.max(dim=-1)[0]
        distance_map = distance_map.clamp_min(0.0)

        if not self.has_trained_prompt_query:
            return distance_map, None

        learned_map, learned_score = self.prompt_query_head(joint_tokens)
        prompt_map = 0.5 * learned_map + 0.5 * distance_map
        return prompt_map, learned_score

    def maps_to_heatmap(self, maps):
        anomaly_map = torch.stack(maps, dim=0).mean(dim=0)
        grid_h, grid_w = self.grid_size
        anomaly_map = anomaly_map.reshape(anomaly_map.shape[0], 1, grid_h, grid_w)
        return F.interpolate(
            anomaly_map,
            size=(self.out_size_h, self.out_size_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)

    def forward(self, images):
        _, patch_tokens = self.encode_image(images)
        textual_map = self.text_anomaly_map(patch_tokens, use_visual_adapter=False)
        visual_map = self.text_anomaly_map(patch_tokens, use_visual_adapter=True)
        maps = [textual_map, visual_map]

        prompt_map, _ = self.prompt_query_map(patch_tokens)
        if prompt_map is not None:
            maps.append(prompt_map)

        return self.maps_to_heatmap(maps)


class AdaptCLIP:
    def __init__(
        self,
        category=None,
        device="cpu",
        backbone="ViT-L-14-336",
        pretrained_dataset="openai",
        out_size_h=256,
        out_size_w=256,
        img_resize=336,
        img_cropsize=336,
        precision=None,
        batch_size=1,
        checkpoint_path=None,
        use_prompt_query=True,
    ):
        self.category = category
        self.device = device
        self.batch_size = batch_size
        self.img_resize = img_resize
        self.img_cropsize = img_cropsize
        self.use_prompt_query = use_prompt_query

        if precision is None:
            precision = "fp16" if str(device).startswith("cuda") else "fp32"

        self.model = AdaptCLIPModel(
            out_size_h=out_size_h,
            out_size_w=out_size_w,
            device=device,
            backbone=backbone,
            pretrained_dataset=pretrained_dataset,
            precision=precision,
            checkpoint_path=checkpoint_path,
        )
        self.model.eval()
        self._is_fit = False

        self.transform = transforms.Compose([
            transforms.Resize((img_resize, img_resize), Image.BICUBIC),
            transforms.CenterCrop(img_cropsize),
            _convert_to_rgb,
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_train, std=std_train),
        ])

    def _infer_category(self, dataset):
        if self.category is not None:
            return self.category
        if hasattr(dataset, "category"):
            return dataset.category
        if hasattr(dataset, "dataset") and hasattr(dataset.dataset, "category"):
            return dataset.dataset.category
        raise ValueError("AdaptCLIP needs category. Pass AdaptCLIP(category='cable', ...) or use a dataset with .category.")

    def _ensure_batch(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(0)
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        return x

    def _to_clip_input(self, x):
        x = self._ensure_batch(x).float().to(self.device)
        imagenet_mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        imagenet_std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        clip_mean = torch.tensor(mean_train, device=x.device).view(1, 3, 1, 1)
        clip_std = torch.tensor(std_train, device=x.device).view(1, 3, 1, 1)

        x = x * imagenet_std + imagenet_mean
        x = x.clamp(0, 1)
        if x.shape[-2:] != (self.img_resize, self.img_resize):
            x = F.interpolate(x, size=(self.img_resize, self.img_resize), mode="bilinear", align_corners=False)
        x = TF.center_crop(x, [self.img_cropsize, self.img_cropsize])
        return (x - clip_mean) / clip_std

    def _iter_batches(self, dataset):
        if hasattr(dataset, "get_loader"):
            yield from dataset.get_loader()
            return
        if isinstance(dataset, DataLoader):
            yield from dataset
            return

        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        yield from loader

    def fit(self, dataset):
        category = self._infer_category(dataset)
        self.model.build_text_feature_gallery(category)

        if self.use_prompt_query:
            patch_gallery = []
            global_gallery = []
            with torch.no_grad():
                for img, _ in self._iter_batches(dataset):
                    img = self._to_clip_input(img)
                    global_token, patch_tokens = self.model.encode_image(img)
                    patch_gallery.append(patch_tokens.reshape(-1, patch_tokens.shape[-1]).cpu())
                    global_gallery.append(global_token.cpu())

            if patch_gallery:
                self.model.prompt_patch_gallery = torch.cat(patch_gallery, dim=0)
                self.model.prompt_global_gallery = torch.cat(global_gallery, dim=0)

        self.category = category
        self._is_fit = True

    @torch.no_grad()
    def _predict_batch_map(self, imgs):
        if not self._is_fit:
            raise RuntimeError("Call fit(dataset) before predict().")
        imgs = self._to_clip_input(imgs)
        return self.model(imgs)

    def predict(self, img):
        heatmap = self._predict_batch_map(img)[0]
        score = heatmap.max()
        return score, heatmap

    def predict_batch(self, imgs):
        heatmaps = self._predict_batch_map(imgs)
        scores = heatmaps.flatten(1).max(dim=1)[0]
        return scores, [heatmap for heatmap in heatmaps]
