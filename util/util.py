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
from skimage.metrics import peak_signal_noise_ratio as compare_psnr #Imports the PSNR metric from skimage  to quantify reconstruction quality for images and video subject to lossy compression. Higher is better
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
'???'


def vis(x):#display an image in image viewer of system regardless if tensor or numpy array
    if isinstance(x, torch.Tensor):
        Image.fromarray(tensor2im(x)).show()
    elif isinstance(x, np.ndarray):
        Image.fromarray(x.astype(np.uint8)).show()
    else:
        raise NotImplementedError('vis for type [%s] is not implemented', type(x))


"""tensorboard"""
from tensorboardX import SummaryWriter #for data logging
from datetime import datetime


def get_summary_writer(log_dir): #log_dir:directy to store logs
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    log_dir = os.path.join(log_dir, datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname()) 
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    writer = SummaryWriter(log_dir) #each time you run your experiments, logs are saved in a new directory with current time, date and machine hostname
    return writer



# keeps track of running avergae of metrics
class AverageMeters(object): #all classes inherit from object
    def __init__(self, dic=None, total_num=None):
        #dic: dictionary of metrics
        #total_num:track of the counts of each metric)
        self.dic = dic or {}
        #(mingcv) self.total_num = total_num
        self.total_num = total_num or {}
        

    def update(self, new_dic): # method appends self.dic and updates total_num with new values.
        for key in new_dic:
            if not key in self.dic:
                self.dic[key] = new_dic[key]
                self.total_num[key] = 1
            else:
                self.dic[key] += new_dic[key]
                self.total_num[key] += 1
        # (mingcv)self.total_num += 1

    def __getitem__(self, key): #overriding [], returns average value for a metric
        return self.dic[key] / self.total_num[key]

    def __str__(self):  #overrider str() and print() to show key and values formatted
        keys = sorted(self.keys())
        res = ''
        for key in keys:
            res += (key + ': %.4f' % self[key] + ' | ')
        return res

    def keys(self): #returns all the metric names
        return self.dic.keys()

'''???why _loss'''
def write_loss(writer, prefix, avg_meters, iteration): 
    #writer is the writer object for logging,SummaryWriter instance
    #prefix is keyword added before each metric e.g. "train"  it logs "accuracy": 85.4 as "train/accuracy"
    #avg_meters: A dictionary argument where keys are metric names (like ‘loss’, ‘accuracy’) and values are their corresponding values at the current iteration (e.g., 0.23, 85.4).
    #iteration: An integer argument representing the current iteration or step number in the training or evaluation process.
    for key in avg_meters.keys():
            meter = avg_meters[key] # for each key, we get the corresponding value from the avg_meters dictionary and assign it to the variable meter.
            writer.add_scalar(os.path.join(prefix, key), meter, iteration)
    # writer.add_scalar(): This method call logs the scalar value (metric) to the writer
'''
avg_meters = {
    “loss”: 0.23,
    “accuracy”: 85.4,
    “precision”: 88.1
}
write_loss(writer, “train”, avg_meters, 10)

Logging: train/loss = 0.23 at step 10
Logging: train/accuracy = 85.4 at step 10
Logging: train/precision = 88.1 at step 10

'''
"""(Mingcv)progress bar"""
import socket

# _, term_width = os.popen('stty size', 'r').read().split() 
term_width = 136 #hardcoded terminal width to 136 characters instead of dymeically reading using stty siz
 
TOTAL_BAR_LENGTH = 65. #length of the progress bar.
last_time = time.time() #when progress is finished
begin_time = last_time  # used to calculate the elapsed time for each step and the total process.


def progress_bar(current, total, msg=None):
                                        #current: The current progress count.
                                        #total: The total count that represents 100% progress.
                                        #msg: An optional message to display alongside the progress bar.
    global last_time, begin_time # for persistence in step_time and total_time
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

    #(mingcv) Go back to the center of the bar.
    for i in range(term_width - int(TOTAL_BAR_LENGTH / 2) + 2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current + 1, total))

    if current < total - 1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


def format_time(seconds): #convert seconds to days,hourse,mintures,seconds,milliseconds
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


def parse_args(args): #converts numeric args seperated by commas into a [] of ints
    str_args = args.split(',') #str1,atr2,str3 becomes [a,b,c]
    parsed_args = []
    for str_arg in str_args:
        arg = int(str_arg)
        if arg >= 0:
            parsed_args.append(arg)
    return parsed_args

#In order to overcome vanishing/explosding gradient, Xavier initialization was introduced. It tries to keep variance of all the layers equal but assumes linear actiivation
# Kaiming He initialization, which takes activation function into account. 

def weights_init_kaiming(m): #layer or module in NN
    classname = m.__class__.__name__ # gets the class name of the layer m using its __class__ attribute and __name__ property. 
    if classname.find('Conv') != -1: #returns -1 if the substring is not found, So here if found
        nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')

    #nn.init.kaiming_normal:Fill the input Tensor(m.weight.data) with values using a Kaiming normal distribution.The method is described in Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification - He, K. et al. (2015). The resulting tensor will have values sampled from N(0,std^2) where std=gain/sqrt(fan_mode)
    #a (float) – the negative slope of the rectifier used after this layer only used with Leaky RelU (0-not used)
    #mode: fan_in(default) or fan_out.Choosing 'fan_in' preserves the magnitude of the variance of the weights in the forward pass. 
    #here intialising for relu,preserving variance of weights in forward pass
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
        # here intialising for relu,preserving variance of weights in forward pass
    elif classname.find('BatchNorm') != -1:
        # nn.init.uniform(m.weight.data, 1.0, 0.02)
        m.weight.data.normal_(mean=0, std=math.sqrt(2. / 9. / 64.)).clamp_(-0.025, 0.025)
        nn.init.constant(m.bias.data, 0.0)
        "??? why std=math.sqrt(2. / 9. / 64.)"
        # .clamp_(-0.025, 0.025): Clamps (limits) the values in the tensor to be between -0.025 and 0.025.
        #bias of the batch normalization layer intiliazed to a constant value of 0.0.


def batch_PSNR(img, imclean, data_range): #calculates avg PSNR for 1 batch of images
    #img batch of images to be evaluated with noise
    #imclean: batch of ground truth images no noise
    Img = img.data.cpu().numpy().astype(np.float32) #converts img to float numpy array in cpu
    #- img.data: Detaches the data from the computation graph.
    #cpu(): Transfers the tensor from GPU to CPU if it’s on GPU.
    #numpy(): Converts the tensor to a NumPy array.
    #astype(np.float32): Converts the array type to float32.
    Iclean = imclean.data.cpu().numpy().astype(np.float32) #same for ground truth
    PSNR = 0
    for i in range(Img.shape[0]): #for all batches add the skmetric.psnr()
        PSNR += compare_psnr(Iclean[i, :, :, :], Img[i, :, :, :], data_range=data_range) 
        #data range: maximum - inimum possible values
    return PSNR / Img.shape[0] #average psnr per batch

#calculates avg SSIM for 1 batch of images
def batch_SSIM(img, imclean):
    Img = img.data.cpu().permute(0, 2, 3, 1).numpy().astype(np.float32)
    #converts img to float numpy array in cpu
    #img.data: Detaches the data from the computation graph.
    #cpu(): Transfers the tensor from GPU to CPU if it’s on GPU.
    #numpy(): Converts the tensor to a NumPy array.
    #astype(np.float32): Converts the array type to float32.
    Iclean = imclean.data.cpu().permute(0, 2, 3, 1).numpy().astype(np.float32)
    #permute(0, 2, 3, 1): Changes the order of dimensions from (batch_size, channels, height, width) to (batch_size, height, width, channels). 
    SSIM = 0

    for i in range(Img.shape[0]): #for all batches add the skmetric.ssim()
        SSIM += structural_similarity(Iclean[i, :, :, :], Img[i, :, :, :], win_size=11,
                                      multichannel=True, data_range=1)
        #win_size:The side-length of the sliding window used in comparison. Must be an odd value
        #multichannel deprecated ...equivalent to channel-axis 
        "??? multichannel=True equal to what number in channel-axis"
    return SSIM / Img.shape[0] #average ssim for1 batch


def data_augmentation(image, mode): #augment given image based on mode passed by manipulating numpy array
    out = np.transpose(image, (1, 2, 0))
    if mode == 0:
        #(mingcv)original
        out = out
    elif mode == 1:
        #(mingcv)flip up and down
        out = np.flipud(out)
    elif mode == 2:
        #(mingcv)rotate counterwise 90 degree
        out = np.rot90(out)
    elif mode == 3:
        #(mingcv)rotate 90 degree and flip up and down
        out = np.rot90(out)
        out = np.flipud(out)
    elif mode == 4:
        #(mingcv)rotate 180 degree
        out = np.rot90(out, k=2)
    elif mode == 5:
        #(mingcv)rotate 180 degree and flip
        out = np.rot90(out, k=2)
        out = np.flipud(out)
    elif mode == 6:
        #(mingcv)rotate 270 degree
        out = np.rot90(out, k=3)
    elif mode == 7:
        #(mingcv)rotate 270 degree and flip
        out = np.rot90(out, k=3)
        out = np.flipud(out)
    return np.transpose(out, (2, 0, 1))
