###############################################################################
# Code from
# https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
# Modified the original code so that it also loads images from the current
# directory as well as the subdirectories
###############################################################################

import torch.utils.data as data

from PIL import Image
import os
import os.path

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def read_fns(filename):#returns all lines in a file as a list
    with open(filename) as f:
        fns = f.readlines()
        fns = [fn.strip() for fn in fns] #This strips any leading and trailing whitespace characters (including newlines) from each line in fns and turns result into a list
    return fns


def is_image_file(filename):#checks if file is image based on extensions
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir, fns=None):#makes dataset from directory with files
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    if fns is None:#if no filename list is given, create list of images
        for root, _, fnames in sorted(os.walk(dir)): #walks through the directory tree starting from dir, returning the root directory path, directories, and file names. The results are sorted.
            for fname in fnames:
                if is_image_file(fname):                
                    path = os.path.join(root, fname)
                    images.append(path)
    else:#create list of images from list
        for fname in fns:
            if is_image_file(fname):
                path = os.path.join(dir, fname)
                images.append(path)

    return images


def default_loader(path):#convert image in path to rgb and open it
    return Image.open(path).convert('RGB') #if grayscale or has alpha channel (rgba)
