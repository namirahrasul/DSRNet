#Contains a base class that defines common options used by both training and testing. 

import argparse #standard library for parsing command-line arguments.
import models #creates a module object.
#models.__dict__ is a dictionary that holds the namespace of the models module. This dictionary contains mappings of all names defined within the module e.g. all the symbols (functions, classes, variables, etc.) defined in that module. Stores key and value
model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))
#convention is camelcase for Classes or Exceptions
#__starting names are magic/dunder methods:the methods starting and ending with double underscores ‘__’. They are defined by built-in classes in Python and commonly used for operator overloading
#callalble() means it can be called like a function.
# so left with functioons and callable classes

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter) #formatter_class=how help messages should be formatted  # argparse.ArgumentDefaultsHelpFormatter: help output should include default values for arguments.
        self.initialized = False #flag for tracking  whether the initialize() method has been called to prevent reinitialization
    def initialize(self): # all the arguments added to parser
        # experiment specifics
        self.parser.add_argument('--name', type=str, default=None,
                                 help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--model', type=str, default='errnet_model', help='chooses which model to use.')
        self.parser.add_argument('--dataset', '-d', type=str, default='sirs_dataset',
                                 help='chooses which dataset to use.')
        self.parser.add_argument('--loss', '-l', type=str, default='losses',
                                 help='chooses which loss to use.')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        self.parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
        self.parser.add_argument('--resume_epoch', '-re', type=int, default=None,
                                 help='checkpoint to use. (default: latest')
        self.parser.add_argument('--seed', type=int, default=0, help='random seed to use. Default=0')
        self.parser.add_argument('--supp_eval', action='store_true', help='supplementary evaluation')
        self.parser.add_argument('--start_now', action='store_true', help='supplementary evaluation')
        self.parser.add_argument('--testr', action='store_true', help='test for reflections')
        self.parser.add_argument('--select', type=str, default=None)

        # for setting input
        self.parser.add_argument('--serial_batches', action='store_true',
                                 help='if true, takes images in order to make batches, otherwise takes them randomly')
        self.parser.add_argument('--nThreads', default=8, type=int, help='# threads for loading data')
        self.parser.add_argument('--max_dataset_size', type=int, default=None,
                                 help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')

        #(mingcv) for display
        self.parser.add_argument('--no-log', action='store_true', help='disable tf logger?')
        self.parser.add_argument('--no-verbose', action='store_true', help='disable verbose info?')
        self.parser.add_argument('--display_winsize', type=int, default=256, help='display window size')
        self.parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
        self.parser.add_argument('--display_id', type=int, default=0,
                                 help='window id of the web display (use 0 to disable visdom)')
        #In Visdom, you can create multiple windows to display various types of information, such as images, graphs, or tables, during the training and testing of machine learning models. The display_id specifies which window (by ID) the visualizations should be displayed in.
        self.parser.add_argument('--display_single_pane_ncols', type=int, default=0,
                                 help='if positive, display all images in a single visdom web panel with certain number of images per row.')

        self.initialized = True
