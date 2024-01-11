import numpy as np
import os
import sys
import ntpath
import time
from . import util, html_util
from subprocess import Popen, PIPE
import wandb
import torchvision
import time
import matplotlib.pyplot as plt




class VisualizerWandb():
    """This class includes several functions that can display/save images and print/save logging information to a Wandb page.

    """

    def __init__(self, opt):
        """Initialize the Visualizer class

        Parameters:
            opt -- stores all the experiment flags; needs to be a subclass of BaseOptions
        Step 1: Cache the training/test options
        Step 2: Create an experiment in the wandb page
        """
        self.opt = opt  # cache the option



        self.logger = wandb.init(
            project="DUPLEX",
            config=vars(opt),
            dir=opt.checkpoints_dir,
            name= opt.name+ time.strftime("%Y%m%d-%H%M%S"),
        )

        # create a logging file to store training losses
        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

        self.image_dir = os.path.join(opt.checkpoints_dir, opt.name, 'image_dir')


        if not os.path.exists(self.image_dir):
            os.makedirs(self.image_dir)

    def reset(self):
        pass

    def witness_sample(self, total_iter, sample, label):
        """
        log specific witness sample along training. 
        Such samples come from the validation set and are used to monitor the training process.
        It should have at least one sample per class.

        Parameters:
            total_iter (int) - - the total iteration during training (not reset to 0)
            sample (OrderedDict) - - dictionary of images to display or save
            label (List) - - list of labels for each image
        """
        fig, axs = plt.subplots(nrows=len(label), ncols=len(sample), figsize=(len(sample)*4, len(label)*4))
        for j, (k, v) in enumerate(sample.items()):
            assert len(v) == len(label)
            for i in range(len(label)):
                axs[i, j].imshow(util.tensor2im(v[i,None]))
                axs[i, j].axis('off')
                if i == 0 :
                    if j == 0:
                        axs[i, j].set_title(k + "     " + str(label[i]), fontsize=20)
                    else:
                        axs[i, j].set_title(k, fontsize=20)
                if j == 0:
                    axs[i, j].set_title(label[i], fontsize=20)

        self.logger.log({"witness": [wandb.Image(fig)]}, step=total_iter)
        plt.close(fig)
        

    def display_current_results(self, visuals, epoch, save_result, total_iter):
        """
        log images to wandb (Max storage of 64 images)

        Parameters:
            visuals (OrderedDict) - - dictionary of images to display or save
            epoch (int) - - the current epoch
            save_result (bool) - - if save the current results to an HTML file
            total_iter (int) - - the total iteration during training (not reset to 0)

        """

        for label, image in visuals.items():
            if save_result :
                image_numpy = util.tensor2im(image)
                img_path = os.path.join(self.image_dir, 'epoch%.3d_iter_%d_%s.png' % (epoch, total_iter, label))
                util.save_image(image_numpy, img_path)
            tosave = image[:64]
            grid_image = torchvision.utils.make_grid(tosave)
            self.logger.log({label: [wandb.Image(grid_image)]}, step=total_iter)

        



    def plot_current_losses(self, epoch, iter, counter_ratio, losses):
        """Store loss, not really useful since I store in wandb, just to maintain consistency with the original codeks

        Parameters:
            epoch (int)           -- current epoch
            counter_ratio (float) -- progress (percentage) in the current epoch, between 0 to 1
            losses (OrderedDict)  -- training losses stored in the format of (name, float) pairs
        """
        if not hasattr(self, 'plot_data'):
            self.plot_data = {'X': [], 'Y': [], 'legend': list(losses.keys())}
        self.plot_data['X'].append(epoch + counter_ratio)
        self.plot_data['Y'].append([losses[k] for k in self.plot_data['legend']])
        
        

    # losses: same format as |losses| of plot_current_losses
    def print_current_losses(self, epoch, iters, losses, t_comp, t_data, total_iter, aux_infos=None, prefix="train/"):
        """print current losses on console; log to wandb; also save the losses to the disk

        Parameters:
            epoch (int) -- current epoch
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            total_iter (int) -- total training iteration (not reset to 0)
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
            t_comp (float) -- computational time per data point (normalized by batch_size)
            t_data (float) -- data loading time per data point (normalized by batch_size)
        """
        message = prefix + 'epoch: %d, iters: %d, total_iter:%d, time: %.3f, data: %.3f, ' % (epoch, iters, total_iter, t_comp, t_data)
        for k, v in losses.items():
            self.logger.log({prefix+k: v}, step=total_iter)
            message += '%s: %.3f, ' % (k, v)
        if aux_infos is not None:
            for k, v in aux_infos.items():
                message += '{}: {}, '.format(k, v)

        print(message)  # print the message
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)  # save the message

