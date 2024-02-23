"""General-purpose training script for DupLEX trainers.

This script works for various trainers (with option '--trainer': e.g., pathwise) and
different datasets (with option '--dataset_mode': e.g., mnistduck).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and trainer ('--trainer').

It first creates trainer, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save trainers.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a pathwise trainer:
        python train.py --dataroot /nrs/funke/senetaire/data --name duckmnist --trainer pathwise_selector
   

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import time
from options.test_options import TestOptions
from data import create_dataset
from duplex_trainer import create_trainer
from util.visualizerwandb import VisualizerWandb
from util.util import tensor2im, save_image
import tqdm
import torch
import os
import pandas as pd
import re

if __name__ == '__main__':
    opt = TestOptions().parse()   # get training options
    dataset = create_dataset(opt, split=opt.phase)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.

    print('The number of testing images = %d' % dataset_size)

    print("Creating trainer")
    trainer = create_trainer(opt)      # create a trainer given opt.trainer and other options
    print("trainer created")
    print("Setting up trainer")
    trainer.setup(opt)               # regular setup: load and print networks; create schedulers
    trainer.load_networks(opt.load_epoch)
    print("trainer setup")
    visualizer = VisualizerWandb(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations

    idx_to_class = dataset.dataset.idx_to_class

    if opt.results_dir is None:
        results_dir = os.path.join(opt.checkpoints_dir, opt.name, 'results', opt.phase+"_load_epoch_"+str(opt.load_epoch),)
    else :
        results_dir = opt.results_dir
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    if opt.save_original:
        original_dir = os.path.join(results_dir, 'original')
        if not os.path.exists(original_dir):
            os.makedirs(original_dir)
      

    if opt.save_counterfactual:
        counterfactual_dir = os.path.join(results_dir, 'counterfactual')
        if not os.path.exists(original_dir):
            os.makedirs(original_dir)
       
    
    if opt.save_mask:
        mask_dir = os.path.join(results_dir, 'mask')
        if not os.path.exists(mask_dir):
            os.makedirs(mask_dir)
      
    
    

    with torch.no_grad():
        # Validation every epoch
        dic_loss_aggregate = {}
        pbar = tqdm.tqdm(enumerate(dataset), desc='Validation', total=int(len(dataset)/opt.batch_size))
        
        dic = {'idx': [], 'name': [], 'y': [], 'y_cf': [], 'pred_y_cf': [], }
        dic.update({f'real_y_cf_{j}': [] for j in range(opt.f_theta_output_classes)})
        count = 0
        for i, data in pbar:
            trainer.set_input(data)
            trainer.evaluate()
            
            for k in range(len(trainer.x)):
                target = trainer.y[k].item()
                x = trainer.x[k].unsqueeze(0).cpu()
                x_cf = trainer.x_cf[k].unsqueeze(0).cpu()
                pi = trainer.pi_to_save[k].unsqueeze(0).cpu()
                y = trainer.y[k].unsqueeze(0).cpu()
                y_cf = trainer.y_cf[k].unsqueeze(0).cpu()
                real_y_cf = trainer.real_y_cf[k].unsqueeze(0).cpu()
                dic['idx'].append(count)
                name_x = data['x_path'][k].split('/')[-1]
                aux_dir = os.path.join(opt.phase, idx_to_class[y.item()], idx_to_class[y_cf.item()])
                
                # if path_x_cf is not None : # Ie pre registered counterfactual
                    # new_directory = re.sub('train/\w+/\w+/\d+(_\w+)?\.\w+', '', "/nrs/funke/adjavond/data/duplex/cyclegan/train/0_gaba/1_acetylcholine/1288_train.png")
                x = tensor2im(x, )
                x_cf = tensor2im(x_cf,)
                pi = tensor2im(pi, )
                if opt.save_original:
                    new_dir = os.path.join(original_dir, aux_dir)
                    if not os.path.exists(new_dir):
                        os.makedirs(new_dir)
                    new_path = os.path.join(new_dir, name_x)
                    save_image(x, new_path)
                if opt.save_counterfactual:
                    new_dir = os.path.join(counterfactual_dir, aux_dir)
                    if not os.path.exists(new_dir):
                        os.makedirs(new_dir)
                    new_path = os.path.join(new_dir, name_x)
                    save_image(x_cf, new_path)

                if opt.save_mask:
                    new_dir = os.path.join(mask_dir, aux_dir)
                    if not os.path.exists(new_dir):
                        os.makedirs(new_dir)
                    new_path = os.path.join(new_dir, name_x)
                    save_image(pi, new_path)
                dic['name'].append(name_x)
                dic['y'].append(target)
                dic['y_cf'].append(y_cf.item())
                for j in range(opt.f_theta_output_classes):
                    dic[f'real_y_cf_{j}'].append(real_y_cf[0,j].item())
                count+=1



        df = pd.DataFrame(dic)
        df.to_csv(os.path.join(results_dir, 'results.csv'))

        aggregated_losses = trainer.get_aggregated_losses()
        visualizer.print_current_losses(-1, -1, aggregated_losses, 0, 0, total_iters, prefix='val/', dataloader_size = len(dataset.dataloader), aux_infos=None)
        visualizer.log_current_losses(losses = aggregated_losses, total_iter=total_iters, prefix = 'val/')

      


