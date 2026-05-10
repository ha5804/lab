from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from utils.corruptions import apply_corruption

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
        self.dir = f"data/MVTec/{self.category}/{self.phase}"

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
            root=self.dir,
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
        
