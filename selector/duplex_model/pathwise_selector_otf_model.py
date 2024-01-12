import torch
import torch.nn.functional as F
import itertools
from util.image_pool import ImagePool
from duplex_model.pathwise_selector_model import PathWiseSelectorModel
from duplex_model.mask_distribution import IndependentRelaxedBernoulli
from duplex_model.networks import define_selector
from dapi_networks.network_utils import init_network, run_inference
from starganv2.inference.model import LatentInferenceModel
import numpy as np 


class PathWiseSelectorOTFModel(PathWiseSelectorModel):
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.

    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        PathWiseSelectorModel.modify_commandline_options(parser, is_train)
        """Add new dataset-specific options, and rewrite default values for existing options.
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

        parser.add_argument('--fast_track', action='store_true', help='if specified, simply generate one counterfactual \
                            and leave it unchecked by the classifier.')
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
        PathWiseSelectorModel.__init__(self, opt)
        if self.isTrain:
            self.model_names.append('latent_inference_model')

            self.fast_track = opt.fast_track
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

        if self.fast_track :
            # Simply generate one counterfactual and leave it unchecked by the classifier.
            return self.netlatent_inference_model(x, target).reshape(x.shape)
        # Copy x batch_size_counterfactual_generation times
        x_multiple = x.unsqueeze(0).expand(self.batch_size_counterfactual_generation, *x.shape).flatten(0,1)
        target_multiple = target.unsqueeze(0).expand(self.batch_size_counterfactual_generation, *target.shape).flatten(0,1)

        # Generate batch_size_counterfactual_generation counterfactuals
        xcf = self.netlatent_inference_model(
            x_multiple.to(self.device),
            target_multiple.to(self.device),
        )

        # Evaluate the counterfactuals
        p = self.netf_theta(xcf).softmax(-1).reshape(self.batch_size_counterfactual_generation, x.shape[0], self.opt.f_theta_output_classes)
        xcf = xcf.reshape(self.batch_size_counterfactual_generation, x.shape[0], *xcf.shape[1:])
        xcf_cat = []
        for k in range(x.shape[0]):
            current_p = p[:,k]
            # Get the predictions
            predictions = torch.argmax(current_p, dim=-1)

            indices = torch.where(predictions == target[k])[0]
            if len(indices) == 0:
                index = torch.argmax(current_p[target[k]],)
            else :
                index = np.random.choice(indices.cpu().numpy())
            # Get the indices of the correct predictions
            xcf_cat.append(xcf[index,k,None])

        xcf = torch.cat(xcf_cat, dim=0)
        return xcf

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
        self.x_expanded = self.x.unsqueeze(0).expand(self.opt.mc_sample_z, *self.x.shape)
        self.y = input['y'].to(self.device)
        self.y_expanded = self.y.unsqueeze(0).expand(self.opt.mc_sample_z, *self.y.shape)

        
        if not self.opt.per_sample_counterfactual:
            self.y_cf = torch.randint_like(self.y, 0, self.opt.f_theta_output_classes)
            self.y_cf_expanded = self.y_cf.unsqueeze(0).expand(self.opt.mc_sample_z, *self.y_cf.shape)
            self.x_cf = self.get_counterfactual(self.x, self.y)
            self.x_cf_expanded = self.x_cf.unsqueeze(0).expand(self.opt.mc_sample_z, *self.x.shape)
        else :
            self.y_cf_expanded = torch.randint_like(self.y_expanded, 0, self.opt.f_theta_output_classes)
            self.x_cf_expanded = self.get_counterfactual(self.x_expanded, self.y_expanded).expand(self.opt.mc_sample_z, *self.x.shape)

        assert self.x_cf_expanded.shape == self.x_expanded.shape, "x_cf_expanded and x_expanded should have the\
              same shape, but have {} and {}".format(self.x_cf_expanded.shape, self.x_expanded.shape)
        self.x_cf_expanded = self.x_cf_expanded.to(self.device)
        self.y_cf_expanded = self.y_cf_expanded.to(self.device)

