from os import walk, path
from typing import Any, Tuple
from random import Random
from torch.utils.data import Dataset
from torchvision.datasets import FashionMNIST, ImageNet, CIFAR10
from cv2 import rotate, flip, ROTATE_90_CLOCKWISE, ROTATE_180, ROTATE_90_COUNTERCLOCKWISE
from numpy import array, concatenate, uint8
from PIL import Image


class INTEL(Dataset):
    def __init__(self, root, split: str = 'train', transform=None, target_transform=None, duplicate_ratio=0.0, rotations=False, return_numpy=False, seed=None):
        
        self.directory = "intel"
    
        self.transform = transform
        self.target_transform = target_transform
        self.rotations = rotations

        self.data = []
        self.targets = []

        self.dataset_size = len(self.data)

        self.labels = {
            'buildings' : 0,
            'forest': 1, 
            'glacier' : 2, 
            'mountain' : 3, 
            'sea' : 4, 
            'street' : 5,
        }

        self.random = Random(seed)
        self.return_numpy = return_numpy

        if split not in ['train', 'test']:
            raise "Wrong split specified. Must be either train or test"
        
        if duplicate_ratio < 0:
            raise ValueError("Duplicate ratio must be greater than 0")
        
        images = []
        for label in self.labels.keys():
            filepath = path.join(*[root, self.directory, 'seg_'+split+'/seg_'+split, label])            # print(filepath)
            image_files = list(walk(filepath))[0][2]
            for image_file in image_files:
                image_file_path = path.join(filepath, image_file)
                image = Image.open(image_file_path)
                image = image.resize((32, 32))
                images.append((array(image, dtype=uint8), array(self.labels[label])))            
        
        self.random.shuffle(images)

        self.data = array([x[0] for x in images])
        self.targets = array([x[1] for x in images])


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):      
        img_numpy, target = self.data[index], self.targets[index]

        # Apply rotations to duplicated samples
        if self.rotations and index > self.dataset_size:
            img_numpy = self.random.choice([rotate(img_numpy, ROTATE_90_COUNTERCLOCKWISE), 
                                        rotate(img_numpy, ROTATE_90_CLOCKWISE), 
                                        rotate(img_numpy, ROTATE_180),
                                        flip(img_numpy, 0)])

        img_pil = Image.fromarray(img_numpy)

        if self.transform is not None:
            img_pil = self.transform(img_pil)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.return_numpy:
            return img_pil, target, img_numpy
        return img_pil, target


class CIFAR10(CIFAR10):
    def __init__(self, duplicate_ratio=0, rotations=False, return_numpy=False, seed=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.random = Random(seed)
        self.rotations = rotations
        self.return_numpy = return_numpy
        self.dataset_size = len(self.data)
            
        if duplicate_ratio > 0:

            if rotations < 0 or rotations > 1:
                raise ValueError("Rotations probability must be in the range (0, 1]")
            
            # pick sample from images acording to ratio
            dup_images = []
            dup_targets = []
            while duplicate_ratio > 1:
                dup_images += self.data
                dup_targets += self.targets
                duplicate_ratio -= 1

            
            indx = []
            indx += self.random.sample([x for x in range(len(self.data))], k = int(duplicate_ratio * len(self.data)))

            for i in indx:
                dup_images.append(self.data[i])
                dup_targets.append(self.targets[i])
                    
 
            self.data = concatenate([self.data, dup_images])
            self.targets = concatenate([self.targets, dup_targets])


    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        
        img_numpy, target = self.data[index], self.targets[index]

        # Apply rotations to duplicated samples
        if self.rotations and index > self.dataset_size:
            img_numpy = self.random.choice([rotate(img_numpy, ROTATE_90_COUNTERCLOCKWISE), 
                                        rotate(img_numpy, ROTATE_90_CLOCKWISE), 
                                        rotate(img_numpy, ROTATE_180),
                                        flip(img_numpy, 0)])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img_pil = Image.fromarray(img_numpy)

        if self.transform is not None:
            img_pil = self.transform(img_pil)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.return_numpy is True:
            return img_pil, target, img_numpy
        return img_pil, target
    

class FashionMNIST(FashionMNIST):
    def __init__(self, duplicate_ratio=0, rotations=False, return_numpy=False, seed=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.random = Random(seed)
        self.rotations = rotations
        self.return_numpy = return_numpy
        self.dataset_size = len(self.data)

        if duplicate_ratio > 0:

            if rotations < 0 or rotations > 1:
                raise ValueError("Rotations probability must be in the range (0, 1]")
            
            # pick sample from images acording to ratio
            dup_images = []
            dup_targets = []
            while duplicate_ratio > 1:
                dup_images += self.data
                dup_targets += self.targets
                duplicate_ratio -= 1

            
            indx = []
            indx += self.random.sample([x for x in range(len(self.data))], k = int(duplicate_ratio * len(self.data)))

            for i in indx:
                dup_images.append(self.data[i])
                dup_targets.append(self.targets[i])
                    
 
            self.data = concatenate([self.data, dup_images])
            self.targets = concatenate([self.targets, dup_targets])


    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img_numpy, target = self.data[index], self.targets[index]

        # Apply rotations to duplicated samples
        if self.rotations and index > self.dataset_size:
            img_numpy = self.random.choice([rotate(img_numpy, ROTATE_90_COUNTERCLOCKWISE), 
                                        rotate(img_numpy, ROTATE_90_CLOCKWISE), 
                                        rotate(img_numpy, ROTATE_180),
                                        flip(img_numpy, 0)])

            
        # img = Image.fromarray(img.numpy(), mode="L")
        img_pil = Image.fromarray(img_numpy.numpy(), mode="L")
        if self.transform is not None:
            img_pil = self.transform(img_pil)
        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.return_numpy is True:
            return img_pil, target, img_numpy
        return img_pil, target
    

class ImageNet(ImageNet):
    def __init__(self, duplicate_ratio=0, rotations=False, return_numpy=False, seed=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.random = Random(seed)
        self.rotations = rotations
        self.return_numpy = return_numpy
        self.dataset_size = len(self.samples)        
            
        if duplicate_ratio > 0:

            if rotations < 0 or rotations > 1:
                raise ValueError("Rotations probability must be in the range (0, 1]")
            
            # pick sample from images acording to ratio
            dup_images = []
            dup_targets = []
            while duplicate_ratio > 1:
                dup_images += self.samples
                dup_targets += self.targets
                duplicate_ratio -= 1            
            indx = []
            indx += self.random.sample([x for x in range(len(self.samples))], k = int(duplicate_ratio * len(self.samples)))

            for i in indx:
                dup_images.append(self.samples[i])
                dup_targets.append(self.targets[i]) 
            self.samples += dup_images
            self.targets += dup_targets


    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        img_pil = self.loader(path)

        if self.rotations and index > self.dataset_size:
            transpose_choice = self.random.choice([Image.FLIP_LEFT_RIGHT, 
                                                    Image.ROTATE_90, 
                                                    Image.ROTATE_180,
                                                    Image.ROTATE_270])            
            img_pil = img_pil.transpose(transpose_choice)

        if self.transform is not None:
            img_pil = self.transform(img_pil)
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        if self.return_numpy is True:
            return img_pil, target, array(img_pil)
        return img_pil, target