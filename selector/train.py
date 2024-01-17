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
from duplex_model import create_model
from util.visualizerwandb import VisualizerWandb
import tqdm
import torch

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
    model = create_model(opt)      # create a model given opt.model and other options
    print("Model created")
    print("Setting up model")
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    print("Model setup")
    visualizer = VisualizerWandb(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations


    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch


        if dataset_val is not None: # Evaluation on validation dataset
            with torch.no_grad():
                # Validation every epoch
                dic_loss_aggregate = {}
                pbar = tqdm.tqdm(enumerate(dataset_val), desc='Validation', total=int(len(dataset_val)/opt.batch_size))
                for i, data in pbar:
                    model.set_input(data)
                    model.evaluate()
                aggregated_losses = model.get_aggregated_losses()
                visualizer.print_current_losses(epoch, -1, aggregated_losses, 0, 0, total_iters, prefix='val/', dataloader_size = len(dataset_val.dataloader), aux_infos=None)
                visualizer.log_current_losses(losses = aggregated_losses, total_iter=total_iters, prefix = 'val/')
                visualizer.display_current_results(model.get_current_visuals(), epoch, True, total_iters)
                model.reset_aggregated_losses()

                # Visualize witness samples
                witness_sample, witness_label= dataset_val.dataset.get_witness_sample()
                model.set_input(witness_sample)
                model.evaluate()
                visualizer.witness_sample(total_iters, model.get_current_visuals(), witness_label,)


        pbar_train= tqdm.tqdm(enumerate(dataset), desc='Training',)
        for i, data in pbar_train:  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            # Train
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            # Evaluate on the train dataset
            if total_iters % opt.print_freq == 0 or total_iters%opt.log_freq == 0 : # If log is required
                with torch.no_grad():
                    model.forward_val()
                    model.calculate_batched_loss_val()
                losses = model.get_current_losses()
                if total_iters % opt.print_freq == 0: # Print losses to console
                    visualizer.print_current_losses(epoch, i, losses, 0, 0, total_iters, dataloader_size = len(dataset.dataloader), aux_infos=None)
                if total_iters % opt.log_freq == 0: # Log losses to wandb
                    visualizer.log_current_losses(losses = losses, total_iter=total_iters, prefix = 'train/')
                
                # TODO : Empty cache here to avoid having large saved element ? Need to add some function in the model class
                
                
            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)
                print('saved')


            iter_data_time = time.time()
            total_iters += 1

        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)
            print('saved')

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
        model.update_learning_rate()                     # update learning rates at the end of every epoch.
