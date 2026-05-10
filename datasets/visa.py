from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import glob
import os
from PIL import Image
from utils.corruptions import apply_corruption

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class ViSA:
    def __init__(
        self,
        category,
        phase="Anomaly",
        batch_size=1,
        shuffle=True,
        #if train phase, recommend True else False
        limit=None,
        #for quick experiment
        limit_per_class=None,
        #each category have many class
        corruption=None,
        severity=0,
    ):
        self.category = category
        self.phase = phase
        self.corruption = corruption
        self.severity = severity
        
        self.dir = os.path.join(ROOT_DIR, "data", "Visa", self.category, "Data", "Images", self.phase)

        self.transform = transforms.Compose([
            #the function is compose is a tool that groups multiple executions together
            transforms.Resize((256, 256)),
            #MVTecData is an image of diffrent sizes so a fixed size is needed.
            transforms.Lambda(
                lambda img: apply_corruption(
                    img,
                    corruption=self.corruption,
                    severity=self.severity,
                )
            ),
            transforms.ToTensor(),
            #convert to a tensor for model input
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),#imagenet statistical value
        ])
        

        self.image_paths = sorted(glob.glob(os.path.join(self.dir, "*.JPG")))
        self.classes = ["Normal", "Anomaly"]
        
        if limit is not None:
            self.image_paths = self.image_paths[:limit]
        
        self.loader = DataLoader(
            self,
            batch_size = batch_size,
            shuffle = shuffle
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        label = 0 if self.phase == "Normal" else 1
        return self.transform(image), label

    def get_dataset(self):
        return self

    def get_loader(self):
        return self.loader

    def get_classes(self):
        return self.classes
