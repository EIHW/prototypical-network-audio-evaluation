import os.path as osp
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms


ROOT_PATH = 'data/'
CURRENT_DATA = 'spectrograms/'
class Spec(Dataset):
    def __init__(self, setname):
        csv_path = osp.join(ROOT_PATH, setname + '.csv')
        lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]
        data = []
        label = []
        # lb = -1
        self.wnids = []
        self.class_idx_to_sample_idx = {}
        for idx, l in enumerate(lines):
            name, wnid = l.split(',')
            path = osp.join(ROOT_PATH, CURRENT_DATA, name)
            if wnid =='pla':
                lb=0
            if wnid =='col':
                lb=1
            if wnid=='tri':
                lb=2
            if wnid == 'joi':
                lb = 3

            data.append(path)
            # print(data)
            label.append(lb)

            # self.class_idx_to_sample_idx[lb].append(idx)



        # print(label)
        self.data = data
        self.labels = label
        # print(data)
        # print(label)

        self.transform = transforms.Compose([
            # transforms.Resize(84),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])

        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, label = self.data[i], self.labels[i]
        # print(labels[i])
        spec = self.transform(Image.open(path).convert('RGB'))
        # print(f'{path} {label}')

        # print(image.shape)
        # print(label)
        # spec = spec.to(device)
        # label = label.to(device)
        return spec, label
