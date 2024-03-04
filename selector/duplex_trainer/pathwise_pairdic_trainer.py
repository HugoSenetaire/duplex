import torch
import torch.nn.functional as F
from duplex_trainer.pathwise_trainer import PathWiseTrainer

class PathWisePairDicTrainer(
    PathWiseTrainer,
):
    """
    This class implements the pathwise selector model for the paired dictionary dataset. To use with SynapseGanCFDataset.
    In such a dataset, **all** the counterfactuals are already provided, and the selector is only responsible for selecting the best one.
    """


    def __init__(self, opt):
        """Initialize selector class.
        If training, will initialize and load the classifier to the given checkpoint.
        Similarly, will initialize and load the latent inference model producing counterfactuals.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        PathWiseTrainer.__init__(self, opt)


    def set_input(self, input):
        """Unpack input data from the dataloader and do
        counterfactual generation using the latent inference model. 
        
        If the option --per_sample_counterfactual is specified,
        it will generate counterfactuals per sample of z,
        otherwise it will generate counterfactuals per samples of x.
        
        Parameters:
            input: a dictionary that contains the data itself and its metadata information.
        """
        self.define_nb_sample()

        self.x_path = input['x_path']
        self.x_cf_path = input['x_cf_path']

        self.x = input['x'].to(self.device)
        self.x_expanded = self.x.unsqueeze(0).expand(self.sample_z, *self.x.shape)
        self.y = input['y'].to(self.device)
        self.y_expanded = self.y.unsqueeze(0).expand(self.sample_z, *self.y.shape)
        
        self.x_cf_expanded = input['x_cf'].to(self.device).unsqueeze(0).expand(self.sample_z, *input['x_cf'].shape)
     
        self.y_cf_expanded = torch.randint_like(self.y_expanded, 0, self.opt.f_theta_output_classes)
        while torch.any(self.y_cf_expanded == self.y_expanded):
            self.aux_y_cf_expanded = torch.randint_like(self.y_cf_expanded, 0, self.opt.f_theta_output_classes)
            self.y_cf_expanded = torch.where(self.y_expanded == self.y_cf_expanded, self.aux_y_cf_expanded, self.y_cf_expanded)
        self.index_select = self.y_cf_expanded.reshape(
                                self.sample_z,
                                self.y.shape[0],
                                1,
                                *[1 for _ in self.x.shape[1:]]
                                ).expand(
                                    self.sample_z,
                                    self.y.shape[0],
                                    1,
                                    *self.x.shape[1:]
                                )
        self.x_cf_expanded = torch.gather(self.x_cf_expanded, 2, self.index_select)
        


        self.y_cf_expanded = self.y_cf_expanded.reshape(self.sample_z, *self.y.shape)
        self.x_cf_expanded = self.x_cf_expanded.reshape(self.sample_z, *self.x.shape)
        self.x_cf = self.x_cf_expanded[0] # We will use the first counterfactual for visualization
        self.y_cf = self.y_cf_expanded[0] # We will use the first counterfactual for visualization
            
        assert self.x_cf_expanded.shape == self.x_expanded.shape, "x_cf_expanded and x_expanded should have the\
              same shape, but have {} and {}".format(self.x_cf_expanded.shape, self.x_expanded.shape)
        self.x_cf_expanded = self.x_cf_expanded.to(self.device)
        self.y_cf_expanded = self.y_cf_expanded.to(self.device)

    def set_input_fix(self, input, target_cf):
        """Unpack input data from the dataloader and do
        counterfactual generation using the latent inference model. 
        
        If the option --per_sample_counterfactual is specified,
        it will generate counterfactuals per sample of z,
        otherwise it will generate counterfactuals per samples of x.
        
        Parameters:
            input: a dictionary that contains the data itself and its metadata information.
        """
        self.mc_sample_z = 1
        self.imp_sample_z = 1
        self.sample_z = 1

        self.x_path = input['x_path']
        self.x_cf_path = input['x_cf_path']

        self.x = input['x'].to(self.device)
        self.x_expanded = self.x.unsqueeze(0).expand(self.sample_z, *self.x.shape)

      
        self.x_cf_expanded = input['x_cf'].to(self.device).unsqueeze(0).expand(self.sample_z, *input['x_cf'].shape)

        self.y = input['y'].to(self.device)
        self.y_expanded = self.y.unsqueeze(0).expand(self.sample_z, *self.y.shape)
        self.set_target()

        self.y_cf_expanded = torch.full_like(self.y_expanded, target_cf)
        self.index_select = self.y_cf_expanded.reshape(
                                self.sample_z,
                                self.y.shape[0],
                                1,
                                *[1 for _ in self.x.shape[1:]]
                                ).expand(
                                    self.sample_z,
                                    self.y.shape[0],
                                    1,
                                    *self.x.shape[1:]
                                )
        self.x_cf_expanded = torch.gather(self.x_cf_expanded, 2, self.index_select)


        self.y_cf_expanded = self.y_cf_expanded.reshape(self.sample_z, *self.y.shape)
        self.x_cf_expanded = self.x_cf_expanded.reshape(self.sample_z, *self.x.shape)
        self.x_cf = self.x_cf_expanded[0] # We will use the first counterfactual for visualization
        self.y_cf = self.y_cf_expanded[0] # We will use the first counterfactual for visualization


            
        assert self.x_cf_expanded.shape == self.x_expanded.shape, "x_cf_expanded and x_expanded should have the\
              same shape, but have {} and {}".format(self.x_cf_expanded.shape, self.x_expanded.shape)
        self.x_cf_expanded = self.x_cf_expanded.to(self.device)
        self.y_cf_expanded = self.y_cf_expanded.to(self.device)


