from __future__ import print_function #if running Python2 makes it work like python3

import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import yaml #Imports PyYAML for parsing YAML files.
from PIL import Image #Imports the Python Imaging Library for image processing.
from skimage.metrics import peak_signal_noise_ratio as compare_psnr #Imports the PSNR metric from skimage  to quantify reconstruction quality for images and video subject to lossy compression.
from skimage.metrics import structural_similarity #Imports the SSIM metric from skimage.A full-reference image quality evaluation index that measures image similarity from three aspects: brightness, contrast, and structure.


def get_config(config):#apth to configuration file
    with open(config, 'r') as stream: #opens the file specified by the config variable in read mode and assigned to variable stream
        return yaml.load(stream) #call loads and parses the YAML content from the file object stream and returns the parsed content.


# Converts a Tensor into a Numpy array then desired imtype
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor, imtype=np.uint8):           
    image_numpy = image_tensor[0].cpu().float().numpy() #image_tensor[0]: first image in batch in tensor
                                                        #.cpu() moves tensor to CPU if in GPU
                                                        #.float() converts to float
                                                        #.numpy() converts to numpy
    if image_numpy.shape[0] == 1: #.shape[0] access channels ,1 = grayscale, 3=RGB
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    #np.tile(A,reps):Construct an array by repeating A the number of times given by reps.
    #repeating graysvale image here in 3 channels to get RGB image
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    #np.transpose(image_numpy, (1, 2, 0)):The image is transposed from the shape (C, H, W) to (H, W, C).
    # + 1: shift the pixel intensity range from [-1, 1] to [0, 2].
    # / 2.0: This scales the values to the range [0, 1].
    # * 255.0: This scales the values to the range [0, 255].
    image_numpy = image_numpy.astype(imtype) #numpy array recast to desired imtype
    if image_numpy.shape[-1] == 6: #after transpose -1 is no.channels and if this is 6
        image_numpy = np.concatenate([image_numpy[:, :, :3], image_numpy[:, :, 3:]], axis=1)
        '''???how does this help '''
        #recombines along width 2 grps of 0-2 and 3-5 
        
    if image_numpy.shape[-1] == 7:#if transposed image has 7 channels
        edge_map = np.tile(image_numpy[:, :, 6:7], (1, 1, 3))
        '''???Tiles (repeats) the seventh channel to create a 3-channel edge_map along the channel axis.'''
        image_numpy = np.concatenate([image_numpy[:, :, :3], image_numpy[:, :, 3:6], edge_map], axis=1)
        '''???how does this help '''
        #recombines along width 2 grps of 0-2 and 3-6 
    return image_numpy


def tensor2numpy(image_tensor):
    image_numpy = torch.squeeze(image_tensor).cpu().float().numpy() #.squeeze(): removes all dimensions of size 1 from the shape of image_tensor
    #PyTorch tensors can have singleton dimensions, which are dimensions with size 1. These dimensions might not carry meaningful information but are retained due to tensor operations.
    #proabably rmeove number of batchs since 1 batch in 1 img (batch,channels,height,width)
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
     #np.transpose(image_numpy, (1, 2, 0)):The image is transposed from the shape (C, H, W) to (H, W, C).
    # + 1: shift the pixel intensity range from [-1, 1] to [0, 2].
    # / 2.0: This scales the values to the range [0, 1].
    # * 255.0: This scales the values to the range [0, 255].
    image_numpy = image_numpy.astype(np.float32)
    #numpy array recast to float
    return image_numpy


# Get model list for resume
#retrieve a specific model checkpoint file based on dirname, epoch,key
def get_model_list(dirname, key, epoch=None): 
    #key: main file_name
    #epoch to indicate at which epoch the model was saved.Choosing epoch allows you to continue training from a specific point or to evaluate the model's performance at different stages of training
    if epoch is None:
        return os.path.join(dirname, key + '_latest.pt') # return latest checkpoint by default
    if os.path.exists(dirname) is False: #if directory doesnt exist  no checkpoint files can be retrieved.
        return None

    print(dirname, key)

    #list of generated checkpoints after specific epohs during traininf
    gen_models = [os.path.join(dirname, f) for f in os.listdir(dirname) if
                  os.path.isfile(os.path.join(dirname, f)) and ".pt" in f and 'latest' not in f]
    #isfile():regular file,not a directory or a special file
    #.pt type file with epoch number specifid, latest means epoch=None
    epoch_index = [int(os.path.basename(model_name).split('_')[-2]) for model_name in gen_models if 'latest' not in model_name]
    #os.path.basename(model_name): extracts the filename from the full path.
    #split('_')[-2]: ['dsrnet', 's', 'epoch14.pt']-> s (second last)
    '''??? why not -1 -> to get 14  chatgpt says int(s) will raise VauleError'''
    print('[i] available epoch list: %s' % epoch_index, gen_models)
    i = epoch_index.index(int(epoch))

    return gen_models[i]

# to preprocess a batch of images for compatibility with models pretrained on the ImageNet dataset
def vgg_preprocess(batch):
    # normalize using imagenet mean and std
    #ImageNet normalization helps preprocess images to have zero mean and unit variance across each channel (RGB).
    mean = batch.new(batch.size()) # Pytorch function creates new tensors (mean and std) with the same shape as the input batch.
    std = batch.new(batch.size())
    mean[:, 0, :, :] = 0.485 #assigned to all elements in the first channel of every image in the batch.
    mean[:, 1, :, :] = 0.456 #2nd channel
    mean[:, 2, :, :] = 0.406 #3rd channel
    std[:, 0, :, :] = 0.229 #1st channel
    std[:, 1, :, :] = 0.224 #2nd channel
    std[:, 2, :, :] = 0.225 #3rd channel
    batch = (batch + 1) / 2 #pixel values shifted [-1,1]->[0,2]->[0,1]
    batch -= mean 
    batch = batch / std
    # each pixel value in batch will have zero mean and unit variance 
    return batch



 #print diagnostic information about the gradients of a neural network 
 #useful for inspecting the average magnitude of gradients during training or optimization of a neural network. 
 # It helps in diagnosing potential issues like vanishing or exploding gradients,
def diagnose_network(net, name='network'): #name: Optional name for the network (default is 'network'), used in printing diagnostic information.
    mean = 0.0 #average mean absolute gradient across all parameters that have gradients.
    count = 0 #no. of paramters with gradients
    for param in net.parameters():
        if param.grad is not None: #trainable Parameters without gradients won't contribute to mean.
            #e.g frozen layers, non trainable paramters,bias,batch normalization parameters 
            mean += torch.mean(torch.abs(param.grad.data)) #mean absolute value of the gradient of the current parameter param 
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path): #create a PIL Image object from the numpy array and then saves it using the save() method of the PIL Image object.
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)

#print shape and statistics of a numpy array 
def print_numpy(x, val=True, shp=False): #val is statistics shp is shape
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten() #converts nD to 1D
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str): 
        #if list makes path for each string
        for path in paths:
            mkdir(path)
    else:      #if string makes path
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def set_opt_param(optimizer, key, value):
    for group in optimizer.param_groups:
        group[key] = value


def vis(x):
    if isinstance(x, torch.Tensor):
        Image.fromarray(tensor2im(x)).show()
    elif isinstance(x, np.ndarray):
        Image.fromarray(x.astype(np.uint8)).show()
    else:
        raise NotImplementedError('vis for type [%s] is not implemented', type(x))


"""tensorboard"""
from tensorboardX import SummaryWriter
from datetime import datetime


def get_summary_writer(log_dir):
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    log_dir = os.path.join(log_dir, datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    writer = SummaryWriter(log_dir)
    return writer


class AverageMeters(object):
    def __init__(self, dic=None, total_num=None):
        self.dic = dic or {}
        # self.total_num = total_num
        self.total_num = total_num or {}

    def update(self, new_dic):
        for key in new_dic:
            if not key in self.dic:
                self.dic[key] = new_dic[key]
                self.total_num[key] = 1
            else:
                self.dic[key] += new_dic[key]
                self.total_num[key] += 1
        # self.total_num += 1

    def __getitem__(self, key):
        return self.dic[key] / self.total_num[key]

    def __str__(self):
        keys = sorted(self.keys())
        res = ''
        for key in keys:
            res += (key + ': %.4f' % self[key] + ' | ')
        return res

    def keys(self):
        return self.dic.keys()


def write_loss(writer, prefix, avg_meters, iteration):
    for key in avg_meters.keys():
        meter = avg_meters[key]
        writer.add_scalar(
            os.path.join(prefix, key), meter, iteration)


"""progress bar"""
import socket

# _, term_width = os.popen('stty size', 'r').read().split()
term_width = 136

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time


def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH * current / total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width - int(TOTAL_BAR_LENGTH) - len(msg) - 3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width - int(TOTAL_BAR_LENGTH / 2) + 2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current + 1, total))

    if current < total - 1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


def format_time(seconds):
    days = int(seconds / 3600 / 24)
    seconds = seconds - days * 3600 * 24
    hours = int(seconds / 3600)
    seconds = seconds - hours * 3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes * 60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds * 1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


def parse_args(args):
    str_args = args.split(',')
    parsed_args = []
    for str_arg in str_args:
        arg = int(str_arg)
        if arg >= 0:
            parsed_args.append(arg)
    return parsed_args


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        # nn.init.uniform(m.weight.data, 1.0, 0.02)
        m.weight.data.normal_(mean=0, std=math.sqrt(2. / 9. / 64.)).clamp_(-0.025, 0.025)
        nn.init.constant(m.bias.data, 0.0)


def batch_PSNR(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    PSNR = 0
    for i in range(Img.shape[0]):
        PSNR += compare_psnr(Iclean[i, :, :, :], Img[i, :, :, :], data_range=data_range)
    return PSNR / Img.shape[0]


def batch_SSIM(img, imclean):
    Img = img.data.cpu().permute(0, 2, 3, 1).numpy().astype(np.float32)
    Iclean = imclean.data.cpu().permute(0, 2, 3, 1).numpy().astype(np.float32)
    SSIM = 0

    for i in range(Img.shape[0]):
        SSIM += structural_similarity(Iclean[i, :, :, :], Img[i, :, :, :], win_size=11,
                                      multichannel=True, data_range=1)
    return SSIM / Img.shape[0]


def data_augmentation(image, mode):
    out = np.transpose(image, (1, 2, 0))
    if mode == 0:
        # original
        out = out
    elif mode == 1:
        # flip up and down
        out = np.flipud(out)
    elif mode == 2:
        # rotate counterwise 90 degree
        out = np.rot90(out)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        out = np.rot90(out)
        out = np.flipud(out)
    elif mode == 4:
        # rotate 180 degree
        out = np.rot90(out, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        out = np.rot90(out, k=2)
        out = np.flipud(out)
    elif mode == 6:
        # rotate 270 degree
        out = np.rot90(out, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        out = np.rot90(out, k=3)
        out = np.flipud(out)
    return np.transpose(out, (2, 0, 1))
