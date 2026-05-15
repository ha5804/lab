from pathlib import Path

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from utils.corruptions import apply_corruption

ROOT_DIR = Path(__file__).resolve().parents[1]

class MyData:
    def __init__(
        self,
        category,
        phase="train",
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
        self.dir = ROOT_DIR / "data" / "MVTec" / self.category / self.phase

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
        

        image_dataset = datasets.ImageFolder(
            root=str(self.dir),
            #read root image
            transform=self.transform
            #apply transform
        )#make train dataset
        
        self.classes = image_dataset.classes
        self.dataset = image_dataset
        
        if limit_per_class is not None:
            indices = []
            counts = {class_idx: 0 for class_idx in range(len(image_dataset.classes))}
            for idx, (_, class_idx) in enumerate(image_dataset.samples):
                if counts[class_idx] < limit_per_class:
                    indices.append(idx)
                    counts[class_idx] += 1
            self.dataset = Subset(image_dataset, indices)
        elif limit is not None:
            limit = min(limit, len(image_dataset))
            self.dataset = Subset(image_dataset, range(limit))

        self.loader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def get_dataset(self):
        return self.dataset

    def get_loader(self):
        return self.loader

    def get_classes(self):
        return self.classes

    def _resolve_dataset_index(self, idx):
        if isinstance(self.dataset, Subset):
            return self.dataset.indices[idx]
        return idx

    def get_image_path(self, idx):
        dataset_idx = self._resolve_dataset_index(idx)
        return Path(self.dataset.dataset.samples[dataset_idx][0]) if isinstance(self.dataset, Subset) else Path(self.dataset.samples[dataset_idx][0])

    def get_mask_path(self, idx):
        image_path = self.get_image_path(idx)
        defect_name = image_path.parent.name
        if self.phase != "test" or defect_name == "good":
            return None
        mask_name = f"{image_path.stem}_mask.png"
        mask_path = ROOT_DIR / "data" / "MVTec" / self.category / "ground_truth" / defect_name / mask_name
        return mask_path if mask_path.exists() else None

    def is_anomaly_label(self, label):
        return self.classes[int(label)] != "good"
        
