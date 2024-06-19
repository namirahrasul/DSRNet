from torch.optim.lr_scheduler import _LRScheduler
#Learning rate scheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
#Reduce learning rate when a metric has stopped improving. Models often benefit from reducing the learning rate by a factor of 2-10 once learning stagnates

class GradualWarmupScheduler(_LRScheduler):
    #Starts with a small learning rate and gradually increases it over a few initial epochs or iterations. This is particularly useful in preventing the model from diverging in the initial phase of training and is often used in training deep networks from scratch.

    #Pros: Prevents early divergence; stabilizes training.
    #Cons: Requires tuning of warm-up duration and rate limits.
    #Use Case: Crucial for training deep networks from scratch.
    """(mingcv) Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """ 
    #total_epoch (number of epochs to reach the target learning rate
    #after scheduler: scheduler after warmup

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier < 1.:
            raise ValueError('multiplier should be greater thant or equal to 1.')
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False #initializes a flag to indicate if warm-up is finished.
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self): #get the current learning rates.
        if self.last_epoch > self.total_epoch: #warm is ovcer , target lr reached
            if self.after_scheduler: #plateuLR used
                if not self.finished: #sets the base learning rates for the after_scheduler to be multiplier times the base learning rates and returns rlearningrate of after scheduler.
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs] #self.base_lrs intialized from _LRScheduler
                    self.finished = True #warm up done
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0: #increase lienarly and propertionally to number of epochs run
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else: #increaase lr non linearly
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in   
                    self.base_lrs]       
        "??? why this formula for non linear"
    
    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        #metrics: A value that will be monitored (e.g., validation loss) to decide if the learning rate needs adjustment.
        if epoch is None: #not resuming from a specific epoch
            epoch = self.last_epoch + 1
        self.last_epoch = epoch if epoch != 0 else 1  # (mingcv) ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in
                         self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)
