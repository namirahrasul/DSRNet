from .base_options import BaseOptions #. means look in the same directory(net_options)

#action=store_true: presence of this argument in the command line will set the corresponding variable to True. If the argument is not specified, the variable will be set to False.Opposite of store_false
class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        # (mingcv)for displays
        self.parser.add_argument('--display_freq', type=int, default=100,
                                 help='frequency of showing training results on screen')
        self.parser.add_argument('--update_html_freq', type=int, default=1000,
                                 help='frequency of saving training results to html')
        self.parser.add_argument('--print_freq', type=int, default=100,
                                 help='frequency of showing training results on console')
        self.parser.add_argument('--eval_freq', type=int, default=1, help='frequency of evaluation')
        self.parser.add_argument('--save_freq', type=int, default=1, help='frequency of save eval samples')
        self.parser.add_argument('--no_html', action='store_true',
                                 help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        self.parser.add_argument('--save_epoch_freq', type=int, default=1,
                                 help='frequency of saving checkpoints at the end of epochs')
        self.parser.add_argument('--debug', action='store_true',
                                 help='only do one epoch and displays at each iteration')
        self.parser.add_argument('--finetune', action='store_true',
                                 help='finetune the network using identity inputs and outputs')
        #Fine-tuning is a process where a pre-trained model is further trained on a new dataset or task.
        '''???identity inputs and outputs:'''
        self.parser.add_argument('--if_align', action='store_true', help='if align 4x')
        #adjusting images so that they meet specific criteria such as size, orientation, or position relative to a reference frame. 
        '''??? here 4x what?'''
        self.parser.add_argument('--weight_dir', type=str, default='')
        self.parser.add_argument('--base_dir', type=str, default='/mnt/lmj_ssd/reflection-removal')


        self.parser.add_argument('--nEpochs', '-n', type=int, default=60, help='# of epochs to run') 
        #when defining the Adam optimizer
        self.parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate for adam')
        self.parser.add_argument('--wd', type=float, default=0, help='weight decay for adam')
        self.parser.add_argument('--num_train', type=int, default=-1) #Number of training samples.
        self.parser.add_argument('--r_pixel_weight', '-rw', type=float, default=1.0, help='weight for r_pixel loss')

        # (mingcv)data augmentation
        self.parser.add_argument('--real20_size', type=int, default=420, help='scale images to compat size')
        self.parser.add_argument('--batchSize', '-b', type=int, default=1, help='input batch size')
        self.parser.add_argument('--loadSize', type=str, default='224,336,448', help='scale images to multiple size')
        self.parser.add_argument('--fineSize', type=str, default='224,224', help='then crop to this size')
        self.parser.add_argument('--no_flip', action='store_true',
                                 help='if specified, do not flip the images for data augmentation')
        self.parser.add_argument('--resize_or_crop', type=str, default='resize_and_crop',
                                 help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop]')
        self.parser.add_argument('--debug_eval', action='store_true',
                                 help='if specified, do not flip the images for data augmentation')
        self.parser.add_argument('--graph', action='store_true', help='print graph')
        self.parser.add_argument('--selected', type=str, nargs='+')
        #nargs='+': the argument can take one or more values. 

        # (mingcv)for discriminator
        self.parser.add_argument('--which_model_D', type=str, default='disc_vgg', choices=['disc_vgg', 'disc_patch'])
        self.parser.add_argument('--gan_type', type=str, default='rasgan',
                                 help='gan/sgan : Vanilla GAN; rasgan : relativistic gan')

        # (mingcv) loss weight
        self.parser.add_argument('--unaligned_loss', type=str, default='vgg',
                                 help='learning rate policy: vgg|mse|ctx|ctx_vgg')
        #unaligned_loss: loss function used to measure the difference between predicted and target images that are not perfectly aligned.
        #vgg: Perceptual Loss/vgg loss.Uses features extracted from a pre-trained network (e.g., VGG) to measure differences in high-level features instead of raw pixel values.
        #mse:Mean Squared Error.Typically for aligned data, but might be included for certain aligned parts or just for comparison.
        #ctx: Contextual loss. Focuses on the context and structure rather than pixel-wise differences.When structural similarity more important
        #ctx_vgg: combination of contextual and VGG perceptual loss.
        self.parser.add_argument('--vgg_layer', type=int, default=31, help='vgg layer of unaligned loss')
        #--vgg_layer:a specific layer from the VGG network (a pre-trained deep convolutional neural network) used to extract high-level features from images to compute the perceptual loss. If u use vgg/ctx_vgg in --unaligned_loss
        self.parser.add_argument('--init_lr', type=float, default=1e-2, help='initial learning rate')
        self.parser.add_argument('--fixed_lr', type=float, default=0, help='initial learning rate')
        "??? default=0 becuse of scheduler?"
        self.parser.add_argument('--lambda_gan', type=float, default=0.01, help='weight for gan loss')
        self.parser.add_argument('--lambda_vgg', type=float, default=0.1, help='weight for vgg loss')
        self.parser.add_argument('--lambda_rec', type=float, default=0.2, help='weight for reconstruction loss')

        self.isTrain = True #indicating that these options are for training.
