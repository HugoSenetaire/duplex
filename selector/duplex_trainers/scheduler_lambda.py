from abc import ABC, abstractmethod
import math


def get_scheduler_lambda(scheduler_lambda_type, target_lambda, target_epoch, init_lambda=0):
    """
    Get the scheduler lambda
    """
    if scheduler_lambda_type == 'constant':
        return ConstantSchedulerLambda(target_lambda, target_epoch, init_lambda)
    elif scheduler_lambda_type == 'linear':
        return LinearSchedulerLambda(target_lambda, target_epoch, init_lambda)
    elif scheduler_lambda_type == 'cosine':
        return CosineSchedulerLambda(target_lambda, target_epoch, init_lambda)
    else :
        raise NotImplementedError('Scheduler lambda type %s is not implemented' % scheduler_lambda_type)

class SchedulerLambda(ABC):
    """
    Abstract class for the scheduler lambda
    """

    def __init__(self, target_lambda, target_epoch, init_lambda):
        """
        Constructor
        """
        self.target_lambda = target_lambda
        self.target_epoch = target_epoch
        self.init_lambda = init_lambda
        self.current_epoch = 0
    
    @abstractmethod
    def __call__(self,):
        """
        Scheduler lambda function
        """
        pass

    def step(self, ):
        """
        Scheduler step function
        """
        if self.current_epoch < self.target_epoch:
            self.current_epoch += 1


class ConstantSchedulerLambda(SchedulerLambda):
    """
    Constant scheduler lambda
    """
    def __init__(self, target_lambda, target_epoch, init_lambda):
        super().__init__(target_lambda, target_epoch, init_lambda)

    def __call__(self, ):
        """
        Constant scheduler lambda function
        """
        return self.target_lambda

class LinearSchedulerLambda(SchedulerLambda):
    """
    Linear scheduler lambda
    """
    def __init__(self, target_lambda, target_epoch, init_lambda,):
        super().__init__(target_lambda,  target_epoch, init_lambda=init_lambda)
        self.current_epoch = 0

    def __call__(self, ):
        """
        Linear scheduler lambda function
        """
        return self.current_epoch/self.target_epoch *(self.target_lambda-self.init_lambda) + self.init_lambda

class CosineSchedulerLambda(SchedulerLambda):
    """
    Cosine scheduler lambda
    """
    def __init__(self, target_lambda, target_epoch, init_lambda,):
        super().__init__(target_lambda, target_epoch, init_lambda= init_lambda,)
        self.current_epoch = 0

    def __call__(self, ):
        """
        Cosine scheduler lambda function
        """
        return self.init_lambda + (self.target_lambda - self.init_lambda) *  math.sin(math.pi/2 * self.current_epoch / self.target_epoch)