import torch
import open_clip
from torch.nn import functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import functional as TF

valid_backbones = ['ViT-B-16-plus-240']
valid_pretrained_datasets = ['laion400m_e32']

from torchvision import transforms

mean_train = [0.48145466, 0.4578275, 0.40821073]
std_train = [0.26862954, 0.26130258, 0.27577711]

def _convert_to_rgb(image):
    return image.convert('RGB')


#CLIP Visual encoder 개조
class OpenClipWinVisual(torch.nn.Module):
    def __init__(self, visual):
        super().__init__()
        self.visual = visual
        #open_clip 내부 ViT encoder, 원본 CLIP backbone 저장
        self.grid_size = self._get_grid_size()
        #patch grid 크기 계산, 예를들어 224,224에서 16짜리 patch를 쓰는 vit라면
        #grid는 14일것이다. 즉 patch의 영역을 grid로 표시하는것. spatial 정보
        self.masks = self._build_patch_masks()
        #mask는 multi-sclae 기록용이다. winclip에서는 여러 scale window사용
        self.scale_begin_indx = [0]

    def _get_grid_size(self):
        #현재 vit가 몇 x 몇 patch grid 가지는지 계산
        grid_size = getattr(self.visual, "grid_size", None)
        #getattr은 파이썬 내장함수로, attribute에 접근하는 함수이다.
        #즉, self.visual에 접근하는데 있으면 self.visual.grid_size 가져오고
        # 없으면 None 반환
        if grid_size is not None:
            if isinstance(grid_size, int):
                #이것도 에러 방지용인데, int면 Return해라 라는뜻
                return (grid_size, grid_size)
            return tuple(grid_size)

        image_size = getattr(self.visual, "image_size", 240)
        patch_size = getattr(self.visual, "patch_size", 16)
        #image_size가 vit모델에 따라 크기가다르다.
        #따라서 아래 코드를 통해 정사각형 크기, 직사각형 크기
        #둘다 처리 가능하도록 하는 robust한 코드이다.
        if isinstance(image_size, tuple):
            image_h, image_w = image_size
        else:
            image_h = image_w = image_size
        if isinstance(patch_size, tuple):
            patch_h, patch_w = patch_size
        else:
            patch_h = patch_w = patch_size
        #위에서 설명하였듯, grid는 결국 patch * grid = image 한 sector
        return (image_h // patch_h, image_w // patch_w)

    def _build_patch_masks(self):
        #grid가 15,15라고 생각하자. patch갯수는 당연히 15 * 15 일것이다.
        num_patches = self.grid_size[0] * self.grid_size[1]
        #torch.eye는 대각성분에 1이되도록한다.
        #이때 num_patch가 16이므로, 16,16짜리 대각성분 1인 행렬 반환
        #이때 mask는 또한 각각의 row이다.
        #따라서 각각의 row에서 mask.bool()을 적용하므로, t,f,f,f,...
        #t.f,f,f,f,
        #f,t,f,f,f,f....
        #와 같은 boolen masking 행렬이 생성된다.
        #이는 patch-level anomaly를 수행하기 위한 mask이다.
        return [mask.bool() for mask in torch.eye(num_patches)]

    def forward(self, image):
        tokens = self.encode_patch_tokens(image)
        return [tokens[:, patch_index] for patch_index in range(tokens.shape[1])]

    def encode_patch_tokens(self, image):
        visual = self.visual
        #마찬가지로 위의 backbone을 불러온다.
        #hasattr을 통해 vit구조인지 확인한다.
        if not all(hasattr(visual, name) for name in ["conv1", "cls_token", "positional_embedding", "ln_pre", "transformer", "ln_post"]):
            raise NotImplementedError("This WinCLIP wrapper needs an open_clip VisionTransformer visual backbone.")
        
        #이미디를 patch token으로 변환한다.
        #(B,3,240,240) -> (B, 768, 15, 15)
        x = visual.conv1(image)
        #grid size를 업데이트한다. 현재 patch grid 크기를 추출한다.
        grid_h, grid_w = x.shape[-2:]
        if self.grid_size != (grid_h, grid_w):
            self.grid_size = (grid_h, grid_w)
            self.masks = self._build_patch_masks()
        #(B,C,H,W)에서 (B,N,D)형태로 flatten
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        #ViT는 맨앞에 special token, CLS를 추가한다.
        #visual의 cls_vector을 추출
        cls_token = visual.class_embedding.to(x.dtype)
        #batch크기에 맞게 복제
        cls_token = cls_token + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
        x = torch.cat([cls_token, x], dim=1)
        #x에 위치정보 추가
        x = x + self._positional_embedding(x, grid_h, grid_w)

        patch_dropout = getattr(visual, "patch_dropout", None)
        if patch_dropout is not None:
            x = patch_dropout(x)

        x = visual.ln_pre(x)
        x = x.permute(1, 0, 2)
        x = visual.transformer(x)
        x = x.permute(1, 0, 2)
        x = x[:, 1:]
        x = visual.ln_post(x)
        #Clip embedding space로 projection
        proj = getattr(visual, "proj", None)
        if proj is not None:
            x = x @ proj
        return x

    def _positional_embedding(self, x, grid_h, grid_w):
        pos = self.visual.positional_embedding.to(dtype=x.dtype, device=x.device)
        #마찬가지로 positional 정보 추가
        if pos.shape[0] == x.shape[1]:
            return pos

        class_pos = pos[:1]
        patch_pos = pos[1:]
        old_grid = int(patch_pos.shape[0] ** 0.5)
        patch_pos = patch_pos.reshape(1, old_grid, old_grid, -1).permute(0, 3, 1, 2)
        patch_pos = F.interpolate(patch_pos, size=(grid_h, grid_w), mode="bicubic", align_corners=False)
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(grid_h * grid_w, -1)
        return torch.cat([class_pos, patch_pos], dim=0)


class OpenClipWinModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.visual = OpenClipWinVisual(model.visual)

    def encode_image(self, image):
        return self.visual(image)

    def encode_text(self, text):
        return self.model.encode_text(text)


class OpenClipAD:
    @staticmethod
    def create_model_and_transforms(model_name, pretrained, scales=None, precision="fp32"):
        model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            precision=precision,
        )
        return OpenClipWinModel(model), preprocess_train, preprocess_val

    @staticmethod
    def get_tokenizer(model_name):
        return open_clip.get_tokenizer(model_name)

class WinClipAD(torch.nn.Module):
    def __init__(self, out_size_h, out_size_w, device, backbone, pretrained_dataset, scales, precision='fp32', **kwargs):
        '''

        :param out_size_h:
        :param out_size_w:
        :param device:
        :param backbone:
        :param pretrained_dataset:
        '''
        super(WinClipAD, self).__init__()

        self.out_size_h = out_size_h
        self.out_size_w = out_size_w
        self.precision = precision # fp16: -40% GPU memory (2.8G->1.6G) with slight performance drop

        self.device = device
        self.get_model(backbone, pretrained_dataset, scales)
        self.phrase_form = '{}'

        # version v1: no norm for each of linguistic embedding
        # version v1:    norm for each of linguistic embedding
        self.version = 'V1' # V1:
        # visual textual, textual_visual
        self.fusion_version = 'textual_visual'

        self.transform = transforms.Compose([
            transforms.Resize((kwargs['img_resize'], kwargs['img_resize']), Image.BICUBIC),
            transforms.CenterCrop(kwargs['img_cropsize']),
            _convert_to_rgb,
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_train, std=std_train)])

        self.gt_transform = transforms.Compose([
            transforms.Resize((kwargs['img_resize'], kwargs['img_resize']), Image.NEAREST),
            transforms.CenterCrop(kwargs['img_cropsize']),
            transforms.ToTensor()])

        print(f'fusion version: {self.fusion_version}')

    def get_model(self, backbone, pretrained_dataset, scales):

        assert backbone in valid_backbones
        assert pretrained_dataset in valid_pretrained_datasets

        model, _, _ = OpenClipAD.create_model_and_transforms(model_name=backbone, pretrained=pretrained_dataset, scales=scales, precision = self.precision)
        tokenizer = OpenClipAD.get_tokenizer(backbone)
        model.eval().to(self.device)
 
        self.masks = model.visual.masks
        self.scale_begin_indx = model.visual.scale_begin_indx
        self.model = model
        self.tokenizer = tokenizer
        self.normal_text_features = None
        self.abnormal_text_features = None
        self.grid_size = model.visual.grid_size
        self.visual_gallery = None
        print("self.grid_size",self.grid_size)

    @torch.no_grad()
    def encode_image(self, image: torch.Tensor):

        if self.precision == "fp16":
            image = image.half()
        image_features = self.model.encode_image(image)
        self.masks = self.model.visual.masks
        self.grid_size = self.model.visual.grid_size
        return [f / f.norm(dim=-1, keepdim=True) for f in image_features]

    @torch.no_grad()
    def encode_text(self, text: torch.Tensor):
        text_features = self.model.encode_text(text)
        return text_features
    
    def build_text_feature_gallery(self, category: str):
        #CLIP에 넣을 Prompt를 정상/이상 prompt 정의
        #Winclip 핵심 contribution1.
        template_level_prompts = [
        "a photo of {}",
        "a blurry photo of {}",
        "a close-up photo of {}",
        "a bright photo of {}",
        ]
        state_level_normal_prompts = [
            "flawless {}",
            "perfect {}",
            "unbroken {}",
            "normal {}",
        ]
        state_level_abnormal_prompts = [
            "broken {}",
            "damaged {}",
            "defective {}",
            "corrupted {}",
        ]
        normal_phrases = []
        abnormal_phrases = []
        
        for template_prompt in template_level_prompts:
            #template_level_prompt는 미리 정의한 prompt들이다.
            #예를들어 "a photo of {}" 이고, 해당 template을 여러개 "a blurry photo of {}" 를 미리 설정
            #각각의 template에 대해 아래 두 for문을 통해 normal과 abonormal prompt를 생성
            # normal prompts
            for normal_prompt in state_level_normal_prompts:
                phrase = template_prompt.format(normal_prompt.format(category))
                normal_phrases += [phrase]

            # abnormal prompts
            for abnormal_prompt in state_level_abnormal_prompts:
                phrase = template_prompt.format(abnormal_prompt.format(category))
                abnormal_phrases += [phrase]

        normal_phrases = self.tokenizer(normal_phrases).to(self.device)
        abnormal_phrases = self.tokenizer(abnormal_phrases).to(self.device)

        
        if self.version == "V1":
            normal_text_features = self.encode_text(normal_phrases)
            abnormal_text_features = self.encode_text(abnormal_phrases)
        elif self.version == "V2":
            normal_text_features = []
            for phrase_id in range(normal_phrases.size()[0]):
                normal_text_feature = self.encode_text(normal_phrases[phrase_id].unsqueeze(0))
                normal_text_feature = normal_text_feature/normal_text_feature.norm(dim=-1, keepdim=True)
                normal_text_features.append(normal_text_feature)
            normal_text_features = torch.cat(normal_text_features, 0).half()
            abnormal_text_features = []
            for phrase_id in range(abnormal_phrases.size()[0]):
                abnormal_text_feature = self.encode_text(abnormal_phrases[phrase_id].unsqueeze(0))
                abnormal_text_feature = abnormal_text_feature/abnormal_text_feature.norm(dim=-1, keepdim=True)
                abnormal_text_features.append(abnormal_text_feature)
            abnormal_text_features = torch.cat(abnormal_text_features, 0).half()
        else:
            raise NotImplementedError

        avr_normal_text_features = torch.mean(normal_text_features, dim=0, keepdim=True)
        avr_abnormal_text_features = torch.mean(abnormal_text_features, dim=0, keepdim=True)

        self.avr_normal_text_features = avr_normal_text_features
        self.avr_abnormal_text_features = avr_abnormal_text_features
        self.text_features = torch.cat([self.avr_normal_text_features,
                                        self.avr_abnormal_text_features], dim=0)
        self.text_features /= self.text_features.norm(dim=-1, keepdim=True)

    def build_image_feature_gallery(self, normal_images):

        self.visual_gallery = []
        visual_features = self.encode_image(normal_images)

        for scale_index in range(len(self.scale_begin_indx)):
            if scale_index == len(self.scale_begin_indx) - 1:
                scale_features = visual_features[self.scale_begin_indx[scale_index]:]
            else:
                scale_features = visual_features[self.scale_begin_indx[scale_index]:self.scale_begin_indx[scale_index+1]]

            self.visual_gallery += [torch.cat(scale_features, dim=0)]


    def calculate_textual_anomaly_score(self, visual_features):
        N = visual_features[0].shape[0]
        scale_anomaly_scores = []
        token_anomaly_scores = torch.zeros((N,self.grid_size[0] * self.grid_size[1]))
        token_weights = torch.zeros((N, self.grid_size[0] * self.grid_size[1]))
        for indx, (features, mask) in enumerate(zip(visual_features, self.masks)):
            normality_and_abnormality_score = (100.0 * features @ self.text_features.T).softmax(dim=-1)
            normality_score = normality_and_abnormality_score[:, 0]
            abnormality_score = normality_and_abnormality_score[:, 1]
            normality_score = normality_score.cpu()

            mask = mask.reshape(-1)
            cur_token_anomaly_score = torch.zeros((N, self.grid_size[0] * self.grid_size[1]))
            if self.precision == "fp16":
                cur_token_anomaly_score = cur_token_anomaly_score.half()
            cur_token_anomaly_score[:, mask] = (1. / normality_score).unsqueeze(1)
            # cur_token_anomaly_score[:, mask] = (1. - normality_score).unsqueeze(1)
            cur_token_weight = torch.zeros((N, self.grid_size[0] * self.grid_size[1]))
            cur_token_weight[:, mask] = 1.

            if indx in self.scale_begin_indx[1:]:
                # deal with the first two scales
                token_anomaly_scores = token_anomaly_scores / token_weights
                scale_anomaly_scores.append(token_anomaly_scores)

                # another scale, calculate from scratch
                token_anomaly_scores = torch.zeros((N, self.grid_size[0] * self.grid_size[1]))
                token_weights = torch.zeros((N, self.grid_size[0] * self.grid_size[1]))

            token_weights += cur_token_weight
            token_anomaly_scores += cur_token_anomaly_score

        # deal with the last one
        token_anomaly_scores = token_anomaly_scores / token_weights
        scale_anomaly_scores.append(token_anomaly_scores)

        scale_anomaly_scores = torch.stack(scale_anomaly_scores, dim=0)
        scale_anomaly_scores = torch.mean(scale_anomaly_scores, dim=0)
        scale_anomaly_scores = 1. - 1. / scale_anomaly_scores

        anomaly_map = scale_anomaly_scores.reshape((N, self.grid_size[0], self.grid_size[1])).unsqueeze(1)
        return anomaly_map

    def calculate_visual_anomaly_score(self, visual_features):
        N = visual_features[0].shape[0]
        scale_anomaly_scores = []
        token_anomaly_scores = torch.zeros((N,self.grid_size[0] * self.grid_size[1]))
        token_weights = torch.zeros((N, self.grid_size[0] * self.grid_size[1]))

        cur_scale_indx = 0
        cur_visual_gallery = self.visual_gallery[cur_scale_indx]

        for indx, (features, mask) in enumerate(zip(visual_features, self.masks)):
            normality_score = 0.5 * (1 - (features @ cur_visual_gallery.T).max(dim=1)[0])
            normality_score = normality_score.cpu()

            mask = mask.reshape(-1)
            cur_token_anomaly_score = torch.zeros((N, self.grid_size[0] * self.grid_size[1]))
            if self.precision == "fp16":
                cur_token_anomaly_score = cur_token_anomaly_score.half()
            cur_token_anomaly_score[:, mask] = normality_score.unsqueeze(1)
            # cur_token_anomaly_score[:, mask] = (1. - normality_score).unsqueeze(1)
            cur_token_weight = torch.zeros((N, self.grid_size[0] * self.grid_size[1]))
            cur_token_weight[:, mask] = 1.

            if indx in self.scale_begin_indx[1:]:
                cur_scale_indx += 1
                cur_visual_gallery = self.visual_gallery[cur_scale_indx]
                # deal with the first two scales
                token_anomaly_scores = token_anomaly_scores / token_weights
                scale_anomaly_scores.append(token_anomaly_scores)

                # another scale, calculate from scratch
                token_anomaly_scores = torch.zeros((N, self.grid_size[0] * self.grid_size[1]))
                token_weights = torch.zeros((N, self.grid_size[0] * self.grid_size[1]))

            token_weights += cur_token_weight
            token_anomaly_scores += cur_token_anomaly_score

        # deal with the last one
        token_anomaly_scores = token_anomaly_scores / token_weights
        scale_anomaly_scores.append(token_anomaly_scores)

        scale_anomaly_scores = torch.stack(scale_anomaly_scores, dim=0)
        scale_anomaly_scores = torch.mean(scale_anomaly_scores, dim=0)

        anomaly_map = scale_anomaly_scores.reshape((N, self.grid_size[0], self.grid_size[1])).unsqueeze(1)
        return anomaly_map

    def forward(self, images):

        visual_features = self.encode_image(images)
        textual_anomaly_map = self.calculate_textual_anomaly_score(visual_features)
        if self.visual_gallery is not None:
            visual_anomaly_map = self.calculate_visual_anomaly_score(visual_features)
        else:
            visual_anomaly_map = textual_anomaly_map

        if self.fusion_version == 'visual':
            anomaly_map = visual_anomaly_map
        elif self.fusion_version == 'textual':
            anomaly_map = textual_anomaly_map
        else:
            anomaly_map = 1. / (1. / textual_anomaly_map + 1. / visual_anomaly_map)

        anomaly_map = F.interpolate(anomaly_map, size=(self.out_size_h, self.out_size_w), mode='bilinear', align_corners=False)
        am_np = anomaly_map.squeeze(1).cpu().numpy()

        am_np_list = []

        for i in range(am_np.shape[0]):
            # am_np[i] = gaussian_filter(am_np[i], sigma=4)
            am_np_list.append(am_np[i])

        return am_np_list

    def train_mode(self):
        self.model.train()

    def eval_mode(self):
        self.model.eval()


class WinCLIP:
    def __init__(
        self,
        category=None,
        device="cpu",
        backbone="ViT-B-16-plus-240",
        pretrained_dataset="laion400m_e32",
        scales=(2, 3),
        out_size_h=256,
        out_size_w=256,
        img_resize=256,
        img_cropsize=240,
        precision=None,
        use_visual_gallery=True,
        batch_size=1,
    ):
        self.category = category
        self.device = device
        self.use_visual_gallery = use_visual_gallery
        self.batch_size = batch_size
        self.img_resize = img_resize
        self.img_cropsize = img_cropsize

        if precision is None:
            precision = "fp16" if str(device).startswith("cuda") else "fp32"

        self.model = WinClipAD(
            out_size_h=out_size_h,
            out_size_w=out_size_w,
            device=device,
            backbone=backbone,
            pretrained_dataset=pretrained_dataset,
            scales=list(scales),
            precision=precision,
            img_resize=img_resize,
            img_cropsize=img_cropsize,
        )
        self.model.eval_mode()
        self._is_fit = False

    def _infer_category(self, dataset):
        if self.category is not None:
            return self.category
        if hasattr(dataset, "category"):
            return dataset.category
        if hasattr(dataset, "dataset") and hasattr(dataset.dataset, "category"):
            return dataset.dataset.category
        raise ValueError("WinCLIP needs category. Pass WinCLIP(category='cable', ...) or use a dataset with .category.")

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
        x = (x - clip_mean) / clip_std
        return x

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

        if self.use_visual_gallery:
            galleries = None
            with torch.no_grad():
                for img, _ in self._iter_batches(dataset):
                    img = self._to_clip_input(img)
                    visual_features = self.model.encode_image(img)

                    if galleries is None:
                        galleries = [[] for _ in range(len(self.model.scale_begin_indx))]

                    for scale_index in range(len(self.model.scale_begin_indx)):
                        begin = self.model.scale_begin_indx[scale_index]
                        if scale_index == len(self.model.scale_begin_indx) - 1:
                            scale_features = visual_features[begin:]
                        else:
                            end = self.model.scale_begin_indx[scale_index + 1]
                            scale_features = visual_features[begin:end]
                        galleries[scale_index].append(torch.cat(scale_features, dim=0))

            self.model.visual_gallery = [torch.cat(scale_gallery, dim=0) for scale_gallery in galleries]

        self.category = category
        self._is_fit = True

    @torch.no_grad()
    def _predict_batch_map(self, imgs):
        if not self._is_fit:
            raise RuntimeError("Call fit(dataset) before predict().")

        imgs = self._to_clip_input(imgs)
        visual_features = self.model.encode_image(imgs)
        textual_anomaly_map = self.model.calculate_textual_anomaly_score(visual_features)

        if self.model.visual_gallery is not None:
            visual_anomaly_map = self.model.calculate_visual_anomaly_score(visual_features)
        else:
            visual_anomaly_map = textual_anomaly_map

        if self.model.fusion_version == "visual":
            anomaly_map = visual_anomaly_map
        elif self.model.fusion_version == "textual":
            anomaly_map = textual_anomaly_map
        else:
            anomaly_map = 1. / (1. / textual_anomaly_map + 1. / visual_anomaly_map)

        return F.interpolate(
            anomaly_map,
            size=(self.model.out_size_h, self.model.out_size_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)

    def predict(self, img):
        heatmap = self._predict_batch_map(img)[0]
        score = heatmap.max()
        return score, heatmap

    def predict_batch(self, imgs):
        heatmaps = self._predict_batch_map(imgs)
        scores = heatmaps.flatten(1).max(dim=1)[0]
        return scores, [heatmap for heatmap in heatmaps]
