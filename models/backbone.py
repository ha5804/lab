from torchvision import models
#CNN, ResNET .. vision model import
import open_clip

def get_patchcore_backbone(name="resnet18"):
    if name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    elif name in {"wide_resnet50_2", "wideresnet50"}:
        model = models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V1)
    else:
        raise ValueError(f"Unsupported PatchCore backbone: {name}")
    model.eval()
    #load resnet backbone for patchcore encoder
    #in eval mode, use mid-layer weights
    return model

def get_backbone(name="resnet18"):
    return get_patchcore_backbone(name)

def get_winclip_backbone(name = 'ViT-B-16'):
    model, _, preprocess = open_clip.create_model_and_transforms(
        name,
        pretrained = 'laion2b_s34b_b88k' 
    )
