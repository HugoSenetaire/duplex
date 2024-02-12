import torch
import torch.nn.functional as F
from starganv2.inference.model import LatentInferenceModel
from .pathwise_selector_trainer import PathWiseSelectorTrainer
import numpy as np 


class PathWiseSelectorOTFTrainer(PathWiseSelectorTrainer):
    """
    This class implements the PathWiseSelectorTrainer for on-the-fly counterfactual generation.
    This trainer stores the latent inference model and the counterfactual generation mode.
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        PathWiseSelectorTrainer.modify_commandline_options(parser, is_train)
        """Add new dataset-specific options, and rewrite default values for existing options.
        Here add the options for the latent inference model and the counterfactual generation.
        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.
        Returns:
            the modified parser.
        """
        parser.add_argument('--latent_model_checkpoint_dir', type=str, required=True, help='path to the latent model checkpoint')
        parser.add_argument('--load_iter_latent_mdodel', type=int, default=10000)
        parser.add_argument('--style_dim_latent_model', type=int, default=64)
        parser.add_argument('--latent_dim_latent_model', type=int, default=16)

        parser.add_argument('--counterfactual_mode', type=str, default='fully_random', choices=['fully_random', 'best', 'random'], help='How to generate counterfactuals. \
                            fully_random will simply generate one counterfactual and leave it unchecked by the classifier. \
                            best will generate a batch of counterfactuals and return the one that is the closest to the target. \
                            random will generate a batch of counterfactuals and return a random one that is classified correctly. \
                            If none is classified correctly, it will return the one that is the closest to the target.')
        parser.add_argument('--batch_size_counterfactual_generation', type=int, default=10, help='Number of counterfactuals to generate per input. \
                            If --per_sample_counterfactual is specified, this is the number of counterfactuals to generate per (ie mc and imp) sample.')  
        parser.add_argument('--per_sample_counterfactual', action='store_true', help='if specified, generate counterfactuals per sample, otherwise generate counterfactuals per batch.')

        return parser

    def __init__(self, opt):
        """Initialize selector class.
        If training, will initialize and load the classifier to the given checkpoint.
        Similarly, will initialize and load the latent inference model producing counterfactuals.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        PathWiseSelectorTrainer.__init__(self, opt)
        self.model_names.append('latent_inference_model')

        self.counterfactual_mode = opt.counterfactual_mode
        img_size = opt.f_theta_input_shape[0]
        style_dim = opt.style_dim_latent_model
        latent_dim = opt.latent_dim_latent_model
        num_domains = opt.f_theta_output_classes
        checkpoint_iter = opt.load_iter_latent_mdodel

        self.netlatent_inference_model = LatentInferenceModel(
            checkpoint_dir=opt.latent_model_checkpoint_dir,
            img_size=img_size,
            style_dim=style_dim,
            latent_dim=latent_dim,
            num_domains=num_domains,
            w_hpf=0.0,
        )
        self.netlatent_inference_model.load_checkpoint(checkpoint_iter)
        self.netlatent_inference_model.to(self.device)
        self.batch_size_counterfactual_generation = opt.batch_size_counterfactual_generation

        

    @torch.no_grad()
    def get_counterfactual(self, x, target):
        """
        Tries to find a counterfactual for the given sample, given the target.
        It creates a batch, and returns one of the samples if it is classified correctly.
        if self.fast_track is specified, it will simply generate one counterfactual and leave it unchecked by the classifier.

        params: x: sample
        params: target: target class
        returns: counterfactual
        raises: CounterfactualNotFound
        """

        if self.counterfactual_mode == 'fully_random':
            # Simply generate one counterfactual and leave it unchecked by the classifier.
            xcf = self.netlatent_inference_model(x, target).reshape(x.shape)
            real_y_cf = self.netf_theta(xcf).softmax(-1).reshape(x.shape[0], self.opt.f_theta_output_classes)


        # TODO : If batch_size * batch_size_counterfactual_generation is too big, it will crash with Integer out of range error
        # Need a failsafe for that
        # Copy x batch_size_counterfactual_generation times
        elif self.counterfactual_mode == 'best':
            x_multiple = x.unsqueeze(0).expand(self.batch_size_counterfactual_generation, *x.shape).flatten(0,1)
            target_multiple = target.unsqueeze(0).expand(self.batch_size_counterfactual_generation, *target.shape).flatten(0,1)

            # Generate batch_size_counterfactual_generation counterfactuals
            xcf = self.netlatent_inference_model(
                x_multiple.to(self.device),
                target_multiple.to(self.device),
            )

            # Evaluate the counterfactuals
            p = self.netf_theta(xcf).softmax(-1).reshape(self.batch_size_counterfactual_generation, x.shape[0], self.opt.f_theta_output_classes)
            xcf = xcf.reshape(self.batch_size_counterfactual_generation, x.shape[0], *x.shape[1:])
            target_one_hot = F.one_hot(target, self.opt.f_theta_output_classes).unsqueeze(0).expand(self.batch_size_counterfactual_generation, target.shape[0], self.opt.f_theta_output_classes)        
            indices = torch.argmin((p - target_one_hot).abs().sum(-1), dim=0)
            indices_xcf = indices.reshape(1, x.shape[0], *[1 for _ in range(xcf.dim()-2)]).expand(1, x.shape[0], *xcf.shape[2:])
            indices_p = indices.reshape(1, x.shape[0], *[1 for _ in range(p.dim()-2)]).expand(1, x.shape[0], *p.shape[2:])

            xcf = xcf.gather(0, indices_xcf).reshape(x.shape)
            real_y_cf = p.gather(0, indices_p).reshape(x.shape[0], self.opt.f_theta_output_classes)

        elif self.counterfactual_mode == 'random':
            xcf_cat = []
            real_y_cf = []
            for k in range(x.shape[0]): # TODO: There might be a non stupid way to do this
                current_p = p[:,k]
                
                # Get the predictions
                predictions = torch.argmax(current_p, dim=-1)

                indices = torch.where(predictions == target[k])[0]
                if len(indices) == 0:
                    index = torch.argmax(current_p[target[k]],)
                else :
                    index = np.random.choice(indices.cpu().numpy())
                real_y_cf.append(p[index,k,None])
                # Get the indices of the correct predictions
                xcf_cat.append(xcf[index,k,None])

            xcf = torch.cat(xcf_cat, dim=0).reshape(x.shape)
            real_y_cf = torch.cat(real_y_cf, dim=0).reshape(x.shape[0], self.opt.f_theta_output_classes)
        return xcf, real_y_cf

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


        self.x = input['x'].to(self.device)
        self.x_expanded = self.x.unsqueeze(0).expand(self.sample_z, *self.x.shape)
        self.y = input['y'].to(self.device)
        self.y_expanded = self.y.unsqueeze(0).expand(self.sample_z, *self.y.shape)

        
        if not self.opt.per_sample_counterfactual:
            self.y_cf = torch.randint_like(self.y, 0, self.opt.f_theta_output_classes)
            while torch.any(self.y == self.y_cf):
                self.aux_y_cf = torch.randint_like(self.y, 0, self.opt.f_theta_output_classes)
                self.y_cf = torch.where(self.y == self.y_cf, self.aux_y_cf, self.y_cf)
            assert torch.all(self.y != self.y_cf), "y and y_cf should be different, but are {} and {}".format(self.y, self.y_cf)
            self.y_cf_expanded = self.y_cf.unsqueeze(0).expand(self.sample_z, *self.y_cf.shape)
            self.x_cf, self.real_y_cf = self.get_counterfactual(self.x, self.y)
            self.x_cf_expanded = self.x_cf.unsqueeze(0).expand(self.sample_z, *self.x.shape)
        else :
            self.y_cf_expanded = torch.randint_like(self.y_expanded, 0, self.opt.f_theta_output_classes)
            while torch.any(self.y_cf_expanded == self.y_expanded):
                self.aux_y_cf_expanded = torch.randint_like(self.y_cf_expanded, 0, self.opt.f_theta_output_classes)
                self.y_cf_expanded = torch.where(self.y_expanded == self.y_cf_expanded, self.aux_y_cf_expanded, self.y_cf_expanded)
            self.y_cf_expanded = torch.randint_like(self.y_expanded, 0, self.opt.f_theta_output_classes)
            self.x_cf_expanded, self.real_y_cf_expanded = self.get_counterfactual(self.x_expanded, self.y_expanded)
            self.y_cf_expanded = self.y_cf_expanded.reshape(self.sample_z, *self.y.shape)
            self.x_cf_expanded = self.x_cf_expanded.reshape(self.sample_z, *self.x.shape)
            self.x_cf = self.x_cf_expanded[0]
            self.y_cf = self.y_cf_expanded[0]
            
        assert self.x_cf_expanded.shape == self.x_expanded.shape, "x_cf_expanded and x_expanded should have the\
              same shape, but have {} and {}".format(self.x_cf_expanded.shape, self.x_expanded.shape)
        self.x_cf_expanded = self.x_cf_expanded.to(self.device)
        self.y_cf_expanded = self.y_cf_expanded.to(self.device)

