#configuration based on neural network
#action=store_true: presence of this argument in the command line will set the corresponding variable to True. If the argument is not specified, the variable will be set to False.Opposite of store_false
import sys

from options.base_option import BaseOptions as Base
from util import util
import os
import torch
import numpy as np
import random


class BaseOptions(Base):
    def initialize(self):
        Base.initialize(self)
        # experiment specifics
        self.parser.add_argument('--inet', type=str, default='net_options',
                                 help='chooses which architecture to use for inet.')
        self.parser.add_argument('--weight_path', type=str, default=None, help='pretrained checkpoint to use.')
        #Path to a pretrained checkpoint for initialization.
        self.parser.add_argument('--init_type', type=str, default='edsr',
                                 help='network initialization [normal|xavier|kaiming|orthogonal|uniform]')
        #(mingcv)for network
        self.parser.add_argument('--hyper', action='store_true',
                                 help='if true, augment input with vgg hypercolumn feature')
        # A hypercolumn refers to a vector formed by concatenating the outputs of all feature maps at a particular spatial location across different layers of the VGG network.
        #Benefits: the network can leverage both low-level details and high-level semantic information simultaneously.provides a richer representation of the input image.
        #Implementation: During the forward pass of the neural network:
        #1.Extract feature maps from various layers of the VGG network.
        #2.Concatenate these feature maps spatially (typically by resizing them to a common size).
        #3.Use this concatenated feature vector as an augmented input alongside the original input to the neural network.

        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args() #Stores parsed command-line arguments.
        self.opt.isTrain = self.isTrain  # (mingcv)train or test
        #to use train_options.py or not
        if self.opt.seed == 0:#for reproducibility across modules.
            seed = random.randrange(2 ** 12 - 1)
            self.opt.seed = seed

        torch.backends.cudnn.deterministic = True
        #PyTorch uses CUDA libraries (like cuDNN) for GPU-accelerated computations. Setting torch.backends.cudnn.deterministic to True ensures that cuDNN uses deterministic algorithms. Operations that rely on cuDNN (like certain convolution operations) will produce the same results on the same input data and configuration.
        torch.manual_seed(self.opt.seed)#Ensure that operations like initializing weights in neural networks or shuffling data batches produce the same results 
        np.random.seed(self.opt.seed)  #(mingcv) seed for every module 
        #np.random.seed:seed for the random number generator in NumPy
        random.seed(self.opt.seed)#seed for the built-in Python random number generator
        #gpu_ids: for multigpu system for parallelization e.g. 0,1,2
        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)
        #converting string of gpus ids into a list of ints
        #(mingcv)set gpu ids
        if len(self.opt.gpu_ids) > 0: #if atleast 1 gpu, set first gpu as cuda device 
            torch.cuda.set_device(self.opt.gpu_ids[0])

        args = vars(self.opt)
        #vars() is a built-in function that returns the __dict__ attribute of an object if it exists. 
        # args = vars(self.opt) converts the attributes of self.opt into  a dictionary args for iteration and accessibility
        print('------------ Options -------------')
        for k, v in sorted(args.items()): # prints all keys and values of dict args
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        #(mingcv) save to the disk
        self.opt.name = self.opt.name or '_'.join([self.opt.model]) 
        # '_'.join([self.opt.model]) if self.opt.name is None
        expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        #experiment directory
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n') # write key and values of options in opt.txt
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')

        if self.opt.debug: #debugging mode with small dataset no multithreading,no flipping
            self.opt.display_freq = 20 #display: visual outputs related to training like plots/images
            self.opt.print_freq = 20 #print: training progress information printed in console  
             #after processing 20 batches of images, the training results will be printed/displayed.
            self.opt.nEpochs = 40
            self.opt.max_dataset_size = 100
            self.opt.no_log = False 
            # controls whether logging of training progress or other information is enabled (False means logging is enabled).
            self.opt.nThreads = 0 #operations are executed in a single thread
            self.opt.decay_iter = 0 # iteration count after which learning rate decay might occur.  0 means no decay
            self.opt.serial_batches = True #whether batches of data are processed in serial (one after another) or in parallel during training. serial:each sample is seen exactly once per epoch in the specified order.
            self.opt.no_flip = True # disables flipping of images during data augmentation.

        return self.opt
