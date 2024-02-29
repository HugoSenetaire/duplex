import os
import torch
from collections import OrderedDict
from abc import ABC, abstractmethod
import numpy as np
import time
from .scheduler_parameter import get_scheduler
import pickle as pkl



class BaseTrainer(ABC):
    """This class is an abstract base class (ABC) for trainers.
    To create a subclass, you need to implement the following five functions:
        -- <__init__>:                      initialize the class; first call BaseTrainer.__init__(self, opt).
        -- <set_input>:                     unpack data from dataset and apply preprocessing.
        -- <forward>:                       produce intermediate results.
        -- <calculate_losses>:              calculate losses batched losses
        -- <optimize_parameters>:           calculate gradients, and update network weights.
        -- <modify_commandline_options>:    (optionally) add trainer-specific options and set default options.
    """

    def __init__(self, opt):
        """Initialize the BaseTrainer class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions

        When creating your custom class, you need to implement your own initialization.
        In this fucntion, you should first call <BaseTrainer.__init__(self, opt)>
        Then, you need to define four lists:
            -- self.loss_names (str list):          specify the training losses that you want to plot and save.
            -- self.trainer_names (str list):         specify the images that you want to display and save.
            -- self.visual_names (str list):        define networks used in our training.
            -- self.optimizers (optimizer list):    define and initialize optimizers. You can define one optimizer for each network. If two networks are updated at the same time, you can use itertools.chain to group them. See cycle_gan_trainer.py for an example.
        """
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')  # get device name: CPU or GPU
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name +  time.strftime("%Y%m%d-%H%M%S"))  # save all the checkpoints to save_dir
        
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        # Save opt to a file :
        with open(os.path.join(self.save_dir, 'opt.pkl'), 'wb') as f:
            pkl.dump(opt, f)

        if opt.preprocess != 'scale_width':  # with [scale_width], input images might have different sizes, which hurts the performance of cudnn.benchmark.
            torch.backends.cudnn.benchmark = True
        self.loss_names = []
        self.trainer_names = []
        self.visual_names = []
        self.optimizers = []
        self.image_paths = []
        self.metric = 0  # used for learning rate policy 'plateau'
        self.aggregated = False

    @staticmethod
    def modify_commandline_options(parser, is_train, opt=None):
        """Add new trainer-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions. Used here for commandline option in attributes of trainer
        Returns:
            the modified parser.
        """
        return parser

    @abstractmethod
    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): includes the data itself and its metadata information.
        """
        pass

    @abstractmethod
    def set_input_fix(self, input, target_cf):
        """Unpack input data from the dataloader and get or generate associated target cf.

        Parameters:
            input (dict): includes the data itself and its metadata information.
            target_cf (int): target class for the counterfactual

        """
        pass

    @abstractmethod
    def forward(self):
        """Run forward pass to create all variable required for training; called by both functions <optimize_parameters>."""
        pass


    @abstractmethod
    def forward_val(self):
        """Run forward pass to create all required variable for measurements; called by <evaluate>."""
        pass


    @abstractmethod
    def calculate_batched_loss(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        pass

    def calculate_batched_loss_val(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        pass

    @abstractmethod
    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        pass

    @abstractmethod
    def evaluate(self):
        """Calculate losses per batched and aggregate; called in every eval iteration"""
        pass

    def setup(self, opt):
        """Load and print networks; create schedulers

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        if self.isTrain:
            self.schedulers = [get_scheduler(optimizer, opt) for optimizer in self.optimizers]
        if not self.isTrain or opt.continue_train:
            self.load_networks(opt.epoch)
        self.print_networks(opt.verbose)

    def eval(self):
        """Make trainers eval mode during test time"""
        for name in self.trainer_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.eval()



    def update_learning_rate(self):
        """Update learning rates for all the networks; called at the end of every epoch"""
        for scheduler in self.schedulers:
            if self.opt.lr_policy == 'plateau':
                scheduler.step(self.metric)
            else:
                scheduler.step()

        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

    def get_aux_info(self,):
        """Return a dictionnary with auxiliary information to be logged
        Here learning rate is returned"""
        dic = {}
        return dic

    def get_current_visuals(self):
        """Return visualization images. train.py will display these images with visdom, and save the images to a HTML"""
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                sample = getattr(self, name)  # We just want a single of the samples here, not everything. Hence thhe [0]
                if len(sample.shape) == 5:
                    sample = sample.flatten(0,1)
                visual_ret[name] = sample  # We just want a single of the samples here, not everything. Hence thhe [0]
        return visual_ret

    def get_current_losses(self):
        """Return traning batched losses / errors. train.py will average them and print out these errors on console, and save them to a file"""
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, 'loss_' + name).mean())  # float(...) works for both scalar tensor and float number
        return errors_ret
    
    def get_current_batched_losses(self):
        """Return val batched losses / errors. train.py will average them and print out these errors on console, and save them to a file
        Only difference with get_current_losses is that it doesn't take the mean of the loss, this is handled by aggregate_losses"""
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = getattr(self, 'loss_' + name)
        return errors_ret

    def aggregate_losses(self, losses):
        """Perform a runnning mean on the aggregated loss using the new found loss.

        Parameters:
            losses (dict): dictionary of losses from single batch
        """
        if not hasattr(self, "loss_aggregate"):
            self.loss_aggregate = {}
            for k, v in losses.items():
                self.loss_aggregate[k] = 0
            self.seen_samples = 0

        current_sample_size = losses[list(losses.keys())[0]].shape[0]

        for k, v in losses.items():
            if v.shape[0] != current_sample_size:
                raise ValueError("Losses must have same batch size but found {} and {}".format(v.shape[0], current_sample_size, k))
            
            self.loss_aggregate[k] = (self.loss_aggregate[k] * self.seen_samples + current_sample_size * v.mean()) / (self.seen_samples + current_sample_size)


    def get_aggregated_losses(self):
        """Return aggregated losses"""
        return self.loss_aggregate    
    
    def reset_aggregated_losses(self):
        """Reset aggregated losses"""
        for k in self.loss_aggregate.keys():
            self.loss_aggregate[k] = 0
        self.seen_samples = 0


    def save_networks(self, epoch):
        """Save all the networks to the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.trainer_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, name)

                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    torch.save(net.cpu().state_dict(), save_path)
                    net.cuda(self.gpu_ids[0])
                else:
                    torch.save(net.cpu().state_dict(), save_path)

    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        """Fix InstanceNorm checkpoints incompatibility (prior to 0.4)"""
        key = keys[i]
        if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
            if module.__class__.__name__.startswith('InstanceNorm') and \
               (key == 'num_batches_tracked'):
                state_dict.pop('.'.join(keys))
        else:
            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)

    def load_networks(self, epoch):
        """Load all the networks from the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.trainer_names:
            if isinstance(name, str):
                load_filename = '%s_net_%s.pth' % (epoch, name)
                load_path = os.path.join(self.save_dir, load_filename)
                net = getattr(self, name)
                # if isinstance(net, torch.nn.DataParallel):
                    # net = net.module
                print('loading the trainer from %s' % load_path)
                # if you are using PyTorch newer than 0.4 (e.g., built from
                # GitHub source), you can remove str() on self.device
                state_dict = torch.load(load_path, map_location=str(self.device))
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata

                # patch InstanceNorm checkpoints prior to 0.4
                for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
                    self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
                net.load_state_dict(state_dict)

    def print_networks(self, verbose):
        """Print the total number of parameters in the network and (if verbose) network architecture

        Parameters:
            verbose (bool) -- if verbose: print the network architecture
        """
        print('---------- Networks initialized -------------')
        for name in self.trainer_names:
            if isinstance(name, str):
                net = getattr(self, name)
                num_params = 0
                if net is None:
                    continue
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
