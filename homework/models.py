from inspect import Parameter
import torch
import torch.nn.functional as F

WIDTH = 64
HEIGHT = 64
CHANNELS = 3
# The larger internal layer gives the model higher capacity to learn different examples for each class
INTERNAL_LAYER_1 = 200
CLASSES = 6

# a useful function, didn't use it, instead used torch.gather()
def one_hot(t, classes):
    return F.one_hot(t, num_classes=classes)

def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.normal_(m.weight, std=0.01)
        torch.nn.init.normal_(m.bias, std=0.01)

class ClassificationLoss(torch.nn.Module):
    def forward(self, input, target):
        """
        Your code here

        Compute mean(-log(softmax(input)_label))

        @input:  torch.Tensor((B,C))
        @target: torch.Tensor((B,), dtype=torch.int64)

        @return:  torch.Tensor((,))

        Hint: Don't be too fancy, this is a one-liner
        """       

        # I could use F.cross_entropy() but I figured it was more in the spirit 
        # of implementing this more transparently here
         
        # calculate log and softmax using the classes dimensions   
        sm = -F.log_softmax(input, 1)

        # perform the indicator function selection by only selecting the output of the matching class labels
        logl=torch.gather(sm, 1, torch.reshape(target, (input.shape[0], 1)))

        # calculate the mean of the selected output
        return torch.mean(logl)

class LinearClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        """
        Your code here
        """
        self.layer = torch.nn.Linear(CHANNELS * WIDTH * HEIGHT, CLASSES)        

    def forward(self, x: torch.Tensor):
        """
        Your code here

        @x: torch.Tensor((B,3,64,64))
        @return: torch.Tensor((B,6))
        """
        return self.layer.forward(x.reshape(-1, CHANNELS * WIDTH * HEIGHT))


class MLPClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()

        """
        Your code here
        """
        self.model = torch.nn.Sequential(
            # first linear layer
            torch.nn.Linear(CHANNELS * WIDTH * HEIGHT, INTERNAL_LAYER_1),
            # first activation layer
            torch.nn.ReLU(),
            # second linear layer
            torch.nn.Linear(INTERNAL_LAYER_1, CLASSES),
            # secon activation layer
            torch.nn.ReLU()
        )

        # initialize weights
        self.model.apply(init_weights)
                    
    def forward(self, x):
        """
        Your code here

        @x: torch.Tensor((B,3,64,64))
        @return: torch.Tensor((B,6))
        """
        x_ = x.reshape(-1, CHANNELS * WIDTH * HEIGHT)
        return self.model.forward(x_)        


model_factory = {
    'linear': LinearClassifier,
    'mlp': MLPClassifier,
}


def save_model(model):
    from torch import save
    from os import path
    for n, m in model_factory.items():
        if isinstance(model, m):
            return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), '%s.th' % n))
    raise ValueError("model type '%s' not supported!" % str(type(model)))


def load_model(model):
    from torch import load
    from os import path
    r = model_factory[model]()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), '%s.th' % model), map_location='cpu'))
    return r
