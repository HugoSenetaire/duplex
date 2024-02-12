"""General-purpose training script for DupLEX trainers.

This script works for various trainers (with option '--trainer': e.g., pathwise) and
different datasets (with option '--dataset_mode': e.g., mnistduck).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and trainer ('--trainer').

It first creates trainer, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save trainers.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a pathwise trainer:
        python train.py --dataroot /nrs/funke/senetaireh/data --name duckmnist --trainer pathwise_selector
   

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import time
from options.test_options import TestOptions
from data import create_dataset
from duplex_trainers import create_trainer
from util.visualizerwandb import VisualizerWandb
from util.util import tensor2im, save_image
import tqdm
import torch
import os
import pandas as pd

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

    if opt.results_dir is None:
        results_dir = os.path.join(opt.checkpoints_dir, opt.name, 'results', opt.phase,)
    else :
        results_dir = opt.results_dir
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    original_dir = os.path.join(results_dir, 'original')
    if not os.path.exists(original_dir):
        os.makedirs(original_dir)

    counterfactual_dir = os.path.join(results_dir, 'counterfactual')
    if not os.path.exists(original_dir):
        os.makedirs(original_dir)
    
    mask_dir = os.path.join(results_dir, 'mask')
    if not os.path.exists(mask_dir):
        os.makedirs(mask_dir)

    target_original_dir = {}
    for k in range(opt.f_theta_output_classes):
        target_original_dir[k] = os.path.join(original_dir, str(k))
        if not os.path.exists(target_original_dir[k]):
            os.makedirs(target_original_dir[k])
    
    target_counterfactual_dir = {}
    for k in range(opt.f_theta_output_classes):
        target_counterfactual_dir[k] = os.path.join(counterfactual_dir, str(k))
        if not os.path.exists(target_counterfactual_dir[k]):
            os.makedirs(target_counterfactual_dir[k])

    target_mask_dir = {}
    for k in range(opt.f_theta_output_classes):
        target_mask_dir[k] = os.path.join(mask_dir, str(k))
        if not os.path.exists(target_mask_dir[k]):
            os.makedirs(target_mask_dir[k])


    with torch.no_grad():
        # Validation every epoch
        dic_loss_aggregate = {}
        
        dic = {'idx': [], 'name': [], 'y': [], 'y_cf': [], 'pred_y_cf': [], }
        dic.update({f'real_y_cf_{j}': [] for j in range(opt.f_theta_output_classes)})
        count = 0
        witness_sample, witness_label= dataset.dataset.get_witness_sample()
        trainer.set_input(witness_sample)
        trainer.evaluate()
        visualizer.witness_sample(total_iters, trainer.get_current_visuals(), witness_label,)
        # pbar = tqdm.tqdm(enumerate(dataset), desc='Validation', total=int(len(dataset)/opt.batch_size))

        # for i, data in pbar:
        #     trainer.set_input(data)
        #     trainer.evaluate()
            
        #     for k in range(len(trainer.x)):
        #         target = trainer.y[k].item()
        #         x = trainer.x[k].unsqueeze(0).cpu()
        #         x_cf = trainer.x_cf[k].unsqueeze(0).cpu()
        #         pi = trainer.pi_to_save[k].unsqueeze(0).cpu()
        #         y = trainer.y[k].unsqueeze(0).cpu()
        #         y_cf = trainer.y_cf[k].unsqueeze(0).cpu()
        #         real_y_cf = trainer.real_y_cf[k].unsqueeze(0).cpu()
        #         dic['idx'].append(count)
        #         name_image = f'{count:05d}.png'
        #         x = tensor2im(x, )
        #         x_cf = tensor2im(x_cf,)
        #         pi = tensor2im(pi, )
        #         save_image(x, os.path.join(target_original_dir[target], name_image))
        #         save_image(x_cf, os.path.join(target_counterfactual_dir[target], name_image))
        #         save_image(pi, os.path.join(target_mask_dir[target], name_image))
        #         dic['name'].append(name_image)
        #         dic['y'].append(target)
        #         dic['y_cf'].append(y_cf.item())
        #         dic['pred_y_cf'].append(trainer.real_y_cf[k].argmax(-1).item())
        #         for j in range(opt.f_theta_output_classes):
        #             dic[f'real_y_cf_{j}'].append(real_y_cf[0,j].item())
        #         count+=1


        # df = pd.DataFrame(dic)
        # df.to_csv(os.path.join(results_dir, 'results.csv'))

        # aggregated_losses = trainer.get_aggregated_losses()
        # visualizer.print_current_losses(-1, -1, aggregated_losses, 0, 0, total_iters, prefix='val/', dataloader_size = len(dataset.dataloader), aux_infos=None)
        # visualizer.log_current_losses(losses = aggregated_losses, total_iter=total_iters, prefix = 'val/')

      


