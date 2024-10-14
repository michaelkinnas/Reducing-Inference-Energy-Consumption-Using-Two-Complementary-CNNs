from pickle import load as pload
from typing import Callable, Optional
from torch.utils.data import Dataset
from cv2 import rotate, flip, ROTATE_90_CLOCKWISE, ROTATE_180, ROTATE_90_COUNTERCLOCKWISE
from random import Random
from numpy import array
from os import walk, path
from PIL import Image



class CIFAR10C(Dataset):
    def __init__(self, root: str, transform: Optional[Callable]=None, train=False, return_numpy=False, duplicate_ratio=0.0, transform_prob=0.0, random_seed=None):
        self.images = []        
        self.transform = transform
        self.return_numpy = return_numpy
        self.random = Random(random_seed)

        if not path.exists(root):
            raise FileNotFoundError("The provided file path directory does not exist")
        

        if train:
            batches = ("data_batch_1",
                        "data_batch_2",
                        "data_batch_3",
                        "data_batch_4",
                        "data_batch_5")
        else:
            # batches = ["test_batch"]
            batches = ("test_batch",)

        for batch in batches:
            with open(path.join(root, batch), 'rb') as fo:
                images_dict = pload(fo, encoding='bytes')
                for img, label in zip(images_dict[b'data'], images_dict[b'labels']):
                    self.images.append((self.__CIFAR_to_numpy(img), array(label))) # shape HxWxC


        if duplicate_ratio != 0:
            if duplicate_ratio < 0:
                raise ValueError("Duplicate ratio must be greater than 0")

            # pick sample from images acording to ratio
            dup_images = []
            while duplicate_ratio > 1:
                dup_images += self.images
                duplicate_ratio -= 1
            dup_images += self.random.sample(self.images, k = int(duplicate_ratio * len(self.images)))

            # If transform prob make random rotations
            if transform_prob != 0:
                if transform_prob < 0 or transform_prob > 1:
                    raise ValueError("Transform probability must be in the range (0, 1]")
                    
                for i, (image, label) in enumerate(dup_images):
                    if self.random.random() <= transform_prob:
                        image = self.random.choice([rotate(image, ROTATE_90_COUNTERCLOCKWISE), 
                                                    rotate(image, ROTATE_90_CLOCKWISE), 
                                                    rotate(image, ROTATE_180),
                                                    flip(image, 0)])
                        
                        dup_images[i] = (image, label)
            
            self.images += dup_images

            # shuffle images
            self.random.shuffle(self.images)

    
    def __CIFAR_to_numpy(self, bin_data):
        r = array(bin_data[0:1024]).reshape((32, 32))
        g = array(bin_data[1024:2048]).reshape((32, 32))
        b = array(bin_data[2048:3072]).reshape((32, 32))   
        return array([r,g,b]).transpose((1, 2, 0))


    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):      
        img, target = self.images[idx][0], self.images[idx][1]
        
        if self.transform:
            img = self.transform(img)
        
        if self.return_numpy:
            return img, target, self.images[idx][0]
       
        return img, target



# Use subset
class ImageNetC(Dataset):
    def __init__(self, root: str, transform: Optional[Callable]=None, return_numpy=False, 
                 duplicate_ratio=0.0, transform_prob=0.0, random_seed=None):
        
        self.images = []        
        self.transform = transform
        self.return_numpy = return_numpy
        self.random = Random(random_seed)

        if not path.exists(root):
            raise FileNotFoundError("The provided file path directory does not exist")
        
        #TODO: Test for existing dir but not valid location of imagenet images               

        for i, (fpath, dirs, files) in enumerate(sorted(walk(root))):
            if len(files) == 0:
                continue

            for image in files:
                image = Image.open(f"{fpath+'/'+image}")
                if image.mode not in ['RGB']:
                    image = image.convert('RGB')
                self.images.append((array(image), array(i-1)))


        if duplicate_ratio != 0:
            if duplicate_ratio < 0:
                raise ValueError("Duplicate ratio must be greater than 0")
            
            # pick sample from images acording to ratio
            dup_images = []
            while duplicate_ratio > 1:
                dup_images += self.images
                duplicate_ratio -= 1
            dup_images += self.random.sample(self.images, k = int(duplicate_ratio * len(self.images)))

            # If transform prob make random rotations
            if transform_prob != 0:
                if transform_prob < 0 or transform_prob > 1:
                    raise ValueError("Transform probability must be in the range (0, 1]")
                    
                for i, (image, label) in enumerate(dup_images):
                    if self.random.random() <= transform_prob:
                        image = self.random.choice([rotate(image, ROTATE_90_COUNTERCLOCKWISE), 
                                                    rotate(image, ROTATE_90_CLOCKWISE), 
                                                    rotate(image, ROTATE_180),
                                                    flip(image, 0)])
                        
                        dup_images[i] = (image, label)

            # Append duplicated images to images list
            self.images += dup_images

        # shuffle images
        self.random.shuffle(self.images)


    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img, target = self.images[idx][0], self.images[idx][1]

        if self.transform:
            img = self.transform(img)
        
        if self.return_numpy:
            return img, target, self.images[idx][0]
       
        return img, target