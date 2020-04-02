from __future__ import print_function

import numpy as np
from torchvision.datasets import VisionDataset

from PIL import Image



import os.path
import sys
import os, torch, re, random, numpy, itertools

from netdissect import pbar

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


def is_npy_file(path):
    return path.endswith('.npy') or path.endswith('.NPY')

def is_image_file(path):
    return None != re.search(r'\.(jpe?g|png)$', path, re.IGNORECASE)

def walk_image_files( rootdir, verbose=None):
    print("Walking image files ... ")
    indexfile = '%s.txt' % rootdir
    if os.path.isfile(indexfile):
        print("from index file: ", indexfile)
        basedir = os.path.dirname(rootdir)
        with open(indexfile) as f:
            result = sorted([os.path.join(basedir, line.strip())
                for line in f.readlines()])
            return result
    result = []
    for dirname, _, fnames in sorted(pbar(os.walk(rootdir),
            desc='Walking %s' % os.path.basename(rootdir))):
        for fname in sorted(fnames):
            if is_image_file(fname) or is_npy_file(fname):
                result.append(os.path.join(dirname, fname))
    return result

  
def make_fast_dataset(dir, class_to_idx, extensions=None, is_valid_file=None):
    print("Making fast dataset ... ", dir)
    images = []
    dir = os.path.expanduser(dir)
    for path in walk_image_files(dir):
        key = os.path.splitext(os.path.relpath(path, dir))[0]
        target = os.path.relpath(path, dir).split('/')[0]
        item = (path, class_to_idx[target])
        images.append(item)
    return images


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)
    
def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

    
def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)
    
class QuickImageFolder(VisionDataset):
    """
    Speed up of ImageFolderInstance using ParallelFolder speed up
    ImageFolderInstance from https://github.com/HobbitLong/CMC/blob/master/dataset.py
    ParallelFolderInstance from https://github.com/davidbau/sidedata/blob/sidedata/netdissect/parallelfolder.py
    """
   

    def __init__(self, root, loader=default_loader, extensions=None, transform=None,
                 target_transform=None, is_valid_file=None, two_crop=False):
        super(QuickImageFolder, self).__init__(root, transform=transform,
                                            target_transform=target_transform)
        classes, class_to_idx = self._find_classes(self.root)
        samples = make_fast_dataset(self.root, class_to_idx, extensions, is_valid_file)
        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.root + "\n"
                                "Supported extensions are: " + ",".join(extensions)))

        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]
        self.imgs = self.samples
        self.two_crop = two_crop

    def _find_classes(self, dir):
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target, index) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        image = self.loader(path)
        if self.transform is not None:
            img = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.two_crop:
            img2 = self.transform(image)
            img = torch.cat([img, img2], dim=0)

        return img, target, index


    def __len__(self):
        return len(self.samples)
    
    
    
    
   