
import os
import random
from typing import Callable,List

from PIL import Image
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import albumentations as A

from data.utils import MultiCompose,local_contrast_normalization

def get_all_file_path(file_dir: str, filter_=('.jpg', '.png')) -> list:
    #遍历文件夹下所有的file
    return [os.path.join(maindir,filename) for maindir,_,file_name_list in os.walk(file_dir) \
        for filename in file_name_list \
        if os.path.splitext(filename)[1] in filter_ ]


class HxqDataset(Dataset):
    def __init__(
        self,
        img_dir: str,
        target_transform:Callable= None, 
        img_gt_transform: Callable= None,
        transform:Callable= None,
        all_transform:Callable= None,
        shape=(3,300,300),
        train:bool = True):
        self._hxq_path_list: List[str] = get_all_file_path(img_dir)
        self._hxq_path_list = [
            x for x in self._hxq_path_list if not x.endswith('_label.png')
        ]
        self.target_transform=target_transform
        self.img_gt_transform= img_gt_transform
        self.transform= transform
        self.all_transform = all_transform
        self.shape=shape
        self.get_ps_material()
        self.train=train

    # TODO: 懒得管外在接口了,先脏修改了实验试试
    def get_ps_material(self):
        p_dir="/home/chiebotgpuhq/MyCode/dataset/anomaly_hxq/crack_segmentation_dataset/images"
        mask_dir="/home/chiebotgpuhq/MyCode/dataset/anomaly_hxq/crack_segmentation_dataset/masks"
        p_path_list=get_all_file_path(p_dir)
        self.ps_material_dict={}
        self.p_path_list=[]
        for p_path in p_path_list:
            file_name=os.path.basename(p_path)
            self.ps_material_dict[p_path]=os.path.join(mask_dir,file_name)
            self.p_path_list.append(p_path)

        broken_glass_dir="/home/chiebotgpuhq/MyCode/dataset/anomaly_hxq/glass_broken"
        self.broken_glass_img_path=get_all_file_path(broken_glass_dir)

    def __len__(self):
        return len(self._hxq_path_list)

    def check_gt_exist(self, img_path):
        ext=os.path.basename(img_path).split('.')[-1]
        gt_path = img_path.replace("."+ext, "_label.png")
        return os.path.exists(gt_path), gt_path

    def _generate_artificion(self,img):
        p_path=random.choice(self.p_path_list)
        p=cv2.imread(p_path)
        mask=cv2.imread(self.ps_material_dict[p_path],0)
        # p=cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)

        p=cv2.resize(p,(img.shape[1],img.shape[0]))
        p=255-p
        mask=cv2.resize(mask,(img.shape[1],img.shape[0]))

        aug = A.Compose([A.HistogramMatching([img], p=1, read_fn=lambda x: x,blend_ratio=(0.8,1))])
        p=aug(image=p)['image']
        gt = (mask != 0)
        mask_pos=np.where(mask > 250)
        img[mask_pos]=p[mask_pos]
        gt = gt.astype(np.int)
        return gt,img

    def _get_glass_broken(self,img):
        p_path=random.choice(self.broken_glass_img_path)
        p=cv2.imread(p_path)
        p=cv2.resize(p,(img.shape[1],img.shape[0]))
        gt=np.ones(img.shape[:2])
        return gt,p


    def generate_anomaly_sample(self, img):
        if random.random()< 1.0:
            return self._generate_artificion(img)
        else:
            return self._get_glass_broken(img)


    def __getitem__(self, idx):
        #  正常是0,异常是1
        img = cv2.imread(self._hxq_path_list[idx])
        rate=1
        while min(img.shape[:2]) <max(self.shape):
            rate=max(self.shape)/min(img.shape[:2])
            img=cv2.resize(img,dsize=None,fx=rate,fy=rate,interpolation=cv2.INTER_LINEAR)
        flag, gt_path = self.check_gt_exist(self._hxq_path_list[idx])
        if flag:
            gt = cv2.imread(gt_path, 0)
            gt=cv2.resize(gt,dsize=None,fx=rate,fy=rate,interpolation=cv2.INTER_LINEAR)
            gt = (gt != 0)
            gt = gt.astype(np.int)
        else:
            if random.random() < 0.5 and self.train:
                gt,img = self.generate_anomaly_sample(img)
            else:
                gt = np.zeros(img.shape[:2])
        img=torch.Tensor(img)
        gt=torch.Tensor(gt)
        img=img.permute(2,0,1)
        gt = gt.mul(255).byte() if gt.dtype != torch.uint8 else gt
        img = img.sub(img.min()).div(img.max() - img.min()).mul(
            255).byte() if img.dtype != torch.uint8 else img

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.transpose(0, 2).transpose(0, 1).numpy(),
                              mode='RGB')
        gt = Image.fromarray(gt.squeeze(0).numpy(), mode='L')

        if self.img_gt_transform is not None:
            img, gt = self.img_gt_transform((img, gt))

        if self.transform is not None:
            img = self.transform(img)

        if gt.max() ==0:
            label=0
        else:
            label=1

        return img, label, gt

    def get_original_gtmaps_normal_class(self) -> torch.Tensor:
        if hasattr(self,'orig_gtmaps'):
            return self.orig_gtmaps
        else:
            all_gt=[sample[-1].cpu().numpy() for sample in self]
            self.orig_gtmaps=torch.Tensor(all_gt)
        return self.orig_gtmaps


class HxqDatasetLoader(object):
    enlarge = True  # enlarge dataset by repeating all data samples ten time, speeds up data loading

    def __init__(self,
                root):
        """
        AD dataset for MVTec-AD. If no MVTec data is found in the root directory,
        the data is downloaded and processed to be stored in torch tensors with appropriate size (defined in raw_shape).
        This speeds up data loading at the start of training.
        :param root: root directory where data is found or is to be downloaded to
        :param normal_class: the class considered nominal
        :param preproc: the kind of preprocessing pipeline
        :param nominal_label: the label that marks nominal samples in training. The scores in the heatmaps always
            rate label 1, thus usually the nominal label is 0, s.t. the scores are anomaly scores.
        :param supervise_mode: the type of generated artificial anomalies.
            See :meth:`fcdd.datasets.bases.TorchvisionDataset._generate_artificial_anomalies_train_set`.
        :param noise_mode: the type of noise used, see :mod:`fcdd.datasets.noise_mode`.
        :param oe_limit: limits the number of different anomalies in case of Outlier Exposure (defined in noise_mode)
        :param online_supervision: whether to sample anomalies online in each epoch,
            or offline before training (same for all epochs in this case).
        :param logger: logger
        :param raw_shape: the height and width of the raw MVTec images before passed through the preprocessing pipeline.
        """

        self.root=root
        self.n_classes = 2  # 0: normal, 1: outlier
        self.shape = (3,448, 448)
        # self.raw_shape = (3, ) + (raw_shape, ) * 2

        # different types of preprocessing pipelines, 'lcn' is for using LCN, 'aug{X}' for augmentations
        img_gt_transform, img_gt_test_transform = None, None
        all_transform = []
        img_gt_transform = MultiCompose([
            transforms.Resize(self.shape[1:], Image.NEAREST),
            transforms.RandomChoice([
                transforms.RandomCrop(self.shape[1:], padding=0),
                transforms.Resize(self.shape[1:], Image.NEAREST)
            ]),
            transforms.ToTensor(),
        ])
        img_gt_test_transform = MultiCompose([
            transforms.Resize(self.shape[1:], Image.NEAREST),
            transforms.ToTensor()
        ])
        test_transform = transforms.Compose([
            transforms.Lambda(
                lambda x: local_contrast_normalization(x, scale='l1')),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomChoice([
                transforms.ColorJitter(0.04, 0.04, 0.04, 0.04),
                transforms.ColorJitter(0.005, 0.0005, 0.0005, 0.0005),
            ]),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: (x + torch.randn_like(x).mul(
                np.random.randint(0, 2)).mul(x.std()).mul(0.1)).clamp(
                    0, 1)),
            transforms.Lambda(
                lambda x: local_contrast_normalization(x, scale='l1')),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])


        all_transform = MultiCompose([
            *all_transform,
        ])
        self.train_set = HxqDataset(os.path.join(self.root, 'train'),None,img_gt_transform,transform,all_transform,train=True)
        self.test_set=HxqDataset(os.path.join(self.root, 'test'),None,img_gt_test_transform,test_transform,None,train=False)

