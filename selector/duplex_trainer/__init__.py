"""This package contains modules related to objective functions, optimizations, and network architectures.

To add a custom trainer class called 'dummy', you need to add a file called 'dummy_trainer.py' and define a subclass Dummytrainer inherited from Basetrainer.
You need to implement the following five functions:
    -- <__init__>:                      initialize the class; first call Basetrainer.__init__(self, opt).
    -- <set_input>:                     unpack data from dataset and apply preprocessing.
    -- <forward>:                       produce intermediate results.
    -- <optimize_parameters>:           calculate loss, gradients, and update network weights.
    -- <modify_commandline_options>:    (optionally) add trainer-specific options and set default options.

In the function <__init__>, you need to define four lists:
    -- self.loss_names (str list):          specify the training losses that you want to plot and save.
    -- self.trainer_names (str list):         define networks used in our training.
    -- self.visual_names (str list):        specify the images that you want to display and save.
    -- self.optimizers (optimizer list):    define and initialize optimizers. You can define one optimizer for each network. If two networks are updated at the same time, you can use itertools.chain to group them. See cycle_gan_trainer.py for an usage.

Now you can use the trainer class by specifying flag '--trainer dummy'.
See our template trainer class 'template_trainer.py' for more details.
"""

import importlib
from duplex_trainer.base_trainer import BaseTrainer


def find_trainer_using_name(trainer_name):
    """Import the module "duplex_trainer/[trainer_name]_trainer.py".

    In the file, the class called DatasetNametrainer() will
    be instantiated. It has to be a subclass of Basetrainer,
    and it is case-insensitive.
    """
    trainer_filename = "duplex_trainer." + trainer_name + "_trainer"
    trainerlib = importlib.import_module(trainer_filename)
    trainer = None
    target_trainer_name = trainer_name.replace('_', '') + 'trainer'
    for name, cls in trainerlib.__dict__.items():
        if name.lower() == target_trainer_name.lower() \
           and issubclass(cls, BaseTrainer):
            trainer = cls

    if trainer is None:
        print("In %s.py, there should be a subclass of Basetrainer with class name that matches %s in lowercase." % (trainer_filename, target_trainer_name))
        exit(0)

    return trainer


def get_option_setter(trainer_name):
    """Return the static method <modify_commandline_options> of the trainer class."""
    trainer_class = find_trainer_using_name(trainer_name)
    return trainer_class.modify_commandline_options


def create_trainer(opt):
    """Create a trainer given the option.

    This function warps the class CustomDatasetDataLoader.
    This is the main interface between this package and 'train.py'/'test.py'

    Example:
        >>> from trainers import create_trainer
        >>> trainer = create_trainer(opt)
    """
    trainer = find_trainer_using_name(opt.trainer)
    instance = trainer(opt)
    print("trainer [%s] was created" % type(instance).__name__)
    return instance
