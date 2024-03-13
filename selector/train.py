"""General-purpose training script for DupLEX models.

This script works for various models (with option '--model': e.g., pathwise) and
different datasets (with option '--dataset_mode': e.g., mnistduck).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a pathwise model:
        python train.py --dataroot /nrs/funke/senetaireh/data --name duckmnist --model pathwise_selector
   

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import time
from options.train_options import TrainOptions
from data import create_dataset
from duplex_trainer import create_trainer
from util.visualizerwandb import VisualizerWandb
from util.util import tensor2im, save_image
import tqdm
import torch
import os
import pandas as pd
import numpy as np

if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    dataset = create_dataset(opt, split='train')  # create a dataset given opt.dataset_mode and other options
    try :
        dataset_val = create_dataset(opt, split='val')  # create a dataset given opt.dataset_mode and other options
    except:
        print("No validation dataset found")
        dataset_val = None
    dataset_size = len(dataset)    # get the number of images in the dataset.
    dataset_size_val = len(dataset_val) if dataset_val is not None else 0

    
    print('The number of training images = %d' % dataset_size)
    print('The number of validation images = %d' % dataset_size_val)

    print("Creating model")
    trainer = create_trainer(opt)      # create a trainer given opt.trainer and other options
    print("trainer created")
    print("Setting up trainer")
    trainer.setup(opt)               # regular setup: load and print networks; create schedulers
    print("trainer setup")
    visualizer = VisualizerWandb(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations

    epoch = 0 
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch


        if dataset_val is not None: # Evaluation on validation dataset
            with torch.no_grad():
                trainer.eval()
                # Validation every epoch
                dic_loss_aggregate = {}
                pbar = tqdm.tqdm(enumerate(dataset_val), desc='Validation', total=int(len(dataset_val)/opt.batch_size))
                for i, data in pbar:
                    trainer.set_input(data)
                    trainer.evaluate()
                aggregated_losses = trainer.get_aggregated_losses()
                visualizer.print_current_losses(epoch, -1, aggregated_losses, 0, 0, total_iters, prefix='val/', dataloader_size = len(dataset_val.dataloader), aux_infos=None)
                visualizer.log_current_losses(losses = aggregated_losses, total_iter=total_iters, prefix = 'val/')
                visualizer.display_current_results(trainer.get_current_visuals(), epoch, True, total_iters)
                trainer.reset_aggregated_losses()

                # Visualize witness samples
                witness_sample, witness_label= dataset_val.dataset.get_witness_sample()
                trainer.set_input(witness_sample)
                trainer.evaluate()
                visualizer.witness_sample(total_iters, trainer.get_current_visuals(), witness_label,)


        pbar_train= tqdm.tqdm(enumerate(dataset), desc='Training',)
        for i, data in pbar_train:  # inner loop within one epoch
            trainer.train()
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            # Train
            trainer.set_input(data)         # unpack data from dataset and apply preprocessing
            trainer.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            # Evaluate on the train dataset
            if total_iters % opt.print_freq == 0 or total_iters%opt.log_freq == 0 : # If log is required
                with torch.no_grad():
                    trainer.eval()
                    trainer.forward_val()
                    trainer.calculate_batched_loss_val()
                losses = trainer.get_current_losses()
                if total_iters % opt.print_freq == 0: # Print losses to console
                    visualizer.print_current_losses(epoch, i, losses, 0, 0, total_iters, dataloader_size = len(dataset.dataloader), aux_infos=None)
                if total_iters % opt.log_freq == 0: # Log losses to wandb
                    visualizer.log_current_losses(losses = losses, total_iter=total_iters, prefix = 'train/')
                    visualizer.log_current_losses(trainer.get_aux_info(), total_iter=total_iters, prefix = 'info/')
                
                # TODO : Empty cache here to avoid having large saved element ? Need to add some function in the trainer class
                
                
            if total_iters % opt.save_latest_freq == 0:   # cache our latest trainer every <save_latest_freq> iterations
                print('saving the latest trainer (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                trainer.save_networks(save_suffix)
                print('saved')


            iter_data_time = time.time()
            total_iters += 1

        if epoch % opt.save_epoch_freq == 0:              # cache our trainer every <save_epoch_freq> epochs
            print('saving the trainer at the end of epoch %d, iters %d' % (epoch, total_iters))
            trainer.save_networks('latest')
            trainer.save_networks(epoch)
            print('saved')

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
        trainer.update_learning_rate()                     # update learning rates at the end of every epoch.



    idx_to_class = dataset.dataset.idx_to_class

    results_dir = os.path.join(trainer.save_dir, 'results', f"test_load_epoch_latest")
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
      
    
    

    with torch.no_grad():
        # Validation every epoch
        dic_loss_aggregate = {}
        dataset = create_dataset(opt, split='test')  # create a dataset given opt.dataset_mode and other options
        pbar = tqdm.tqdm(enumerate(dataset), desc='Sample element', total=int(len(dataset.dataset)/opt.batch_size))
        
        dic = {'idx': [], 'name': [], 'y': [], 'y_cf': [], 'pred_y_cf': [], }
        dic.update({f'real_y_cf_{j}': [] for j in range(opt.f_theta_output_classes)})
        count = 0
        trainer.eval()
        trainer.load_networks("latest")
        duplex = trainer.duplex
        for i, data in pbar:
            for key,value in dataset.dataset.idx_to_class.items():
                time_init = time.time()
                trainer.set_input_fix(data, key)
                trainer.evaluate()

                time_init = time.time()
                for k in range(len(trainer.x_expanded[0])):
                    target = trainer.y_expanded[0,k].item()
                    x = trainer.x_expanded[0,k].unsqueeze(0).cpu()
                    x_cf = trainer.x_cf_expanded[0,k].unsqueeze(0).cpu()
                    pi = trainer.pi_to_save[0,k].unsqueeze(0).cpu()
                    y = trainer.y_expanded[0,k].unsqueeze(0).cpu()
                    y_cf = trainer.y_cf_expanded[0,k].unsqueeze(0).cpu()
                    real_y_cf = trainer.y_tilde_pi[0,k].unsqueeze(0).cpu()
                    dic['idx'].append(count)
                    name_x = data['x_path'][k].split('/')[-1]
                    aux_dir = os.path.join("test", idx_to_class[y.item()], idx_to_class[y_cf.item()])
                    
                    x = tensor2im(x, )
                    x_cf = tensor2im(x_cf,)
                    pi = tensor2im(pi, )


                    new_dir = os.path.join(original_dir, aux_dir)
                    if not os.path.exists(new_dir):
                        os.makedirs(new_dir)
                    new_path = os.path.join(new_dir, name_x)
                    save_image(x, new_path)


                    new_dir = os.path.join(counterfactual_dir, aux_dir)
                    if not os.path.exists(new_dir):
                        os.makedirs(new_dir)
                    new_path = os.path.join(new_dir, name_x)
                    save_image(x_cf, new_path)

                    new_dir = os.path.join(mask_dir, aux_dir)
                    if not os.path.exists(new_dir):
                        os.makedirs(new_dir)
                    new_path = os.path.join(new_dir, name_x)
                    save_image(pi, new_path)
                    new_path_numpy = new_path.replace('.png', '.npy')
                    np.save(new_path_numpy, pi)


                    dic['name'].append(name_x)
                    dic['y'].append(target)
                    dic['y_cf'].append(y_cf.item())
                    for j in range(opt.f_theta_output_classes):
                        dic[f'real_y_cf_{j}'].append(real_y_cf[0,j].item())
                    count+=1
