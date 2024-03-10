import os

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from utilities.utils import *

class CelebADataset(Dataset):
    def __init__(self, data_source, data_type, transform=None, target_transform=None):        
        self.data_source = data_source
        self.data_type = data_type
        self.transform = transform
        self.target_transform = target_transform
        df1 = pd.read_csv(os.path.join(data_source, f'list_attr_celeba.csv'))
        all_image_ids, all_labels = df1['image_id'], df1['Male']
        
        df2 = pd.read_csv(os.path.join(data_source, f'list_eval_partition.csv'))
        if data_type == 'train':
            self.image_ids = all_image_ids[df2['partition'] == 0]
            self.labels = all_labels[df2['partition'] == 0]
        elif data_type == 'val':
            self.image_ids = all_image_ids[df2['partition'] == 1]
            self.labels = all_labels[df2['partition'] == 1]
        elif data_type == 'test':
            self.image_ids = all_image_ids[df2['partition'] == 2]
            self.labels = all_labels[df2['partition'] == 2]
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_source, f'celeba/{self.image_ids.iloc[idx]}')
        image = Image.open(img_path).convert('RGB')
        label = max(0, self.labels.iloc[idx])
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        
        return image, label

if __name__ == '__main__':
    logger.info('... [ Testing dataset ] ...')