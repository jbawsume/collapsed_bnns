import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchvision
from skimage import io

class StateFarmDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index,1], self.annotations.iloc[index,2])
        image = io.imread(img_path)
        y_label_str = self.annotations.iloc[index,1]
        y_label = torch.tensor(int(y_label_str[-1]))

        if self.transform:
            #data_norm_mean, data_norm_std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
            data_norm_mean, data_norm_std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
            transform = torchvision.transforms.Compose([
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.Resize(32), #32 for resnet  #128 for linear #64 for conv
                torchvision.transforms.ToTensor(),
                torchvision.transforms.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                #torchvision.transforms.Normalize(mean=data_norm_mean, std=data_norm_std)
            ])
            image = transform(image)

        return (image, y_label)