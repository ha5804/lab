from torchvision import models
#CNN, ResNET .. vision model import
import open_clip
def get_patchcore_backbone(name = "resnet18"):
    model = models.resnet18(weights = models.ResNet18_Weights.IMAGENET1K_V1)
    model.eval()
    #load resnet backbone for patchcore encoder
    #in eval mode, use mid-layer weights
    return model

def get_winclip_backbone(name = 'ViT-B-16'):
    model, _, preprocess = open_clip.create_model_and_transforms(
        name,
        pretrained = 'laion2b_s34b_b88k' 
    )
    