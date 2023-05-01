from .models import ClassificationLoss, load_model, model_factory, save_model
from .utils import accuracy, load_data, LABEL_NAMES
import torch
from torch.optim import Optimizer
from torch.optim import SGD
import torch.utils.tensorboard as tb
import tempfile

LEARNING_RATE = 0.01

logger = tb.SummaryWriter('./test', flush_secs=1)

def train(args):
    
    try:
        model = load_model(args.model)
    except:
        model = model_factory[args.model]()

    # load training data
    training_data = load_data('./data/train')

    # load validation data
    valid_data = load_data('./data/valid')
    
    loss_fn = ClassificationLoss()    
    train_m = model.parameters()
    optimizer = SGD(train_m, args.rate)

    # training data
    Y = None
    X = None
    for data, target in training_data:
        if X is None:
            X = torch.zeros_like(data)
            Y = torch.zeros_like(target)
        else:
            X = torch.cat((X, data))
            Y = torch.cat((Y, target))

    # validation data
    Y_val = None
    X_val = None 
    for data, target in valid_data:
        if X_val is None:
            X_val = torch.zeros_like(data)
            Y_val = torch.zeros_like(target)
        else:
            X_val = torch.cat((X_val, data), 0)
            Y_val = torch.cat((Y_val, target), 0)
        
    for epoch in range(args.epochs):

        # reset gradients
        optimizer.zero_grad()
        
        X_train = X
        Y_train = Y

        # train model
        pred = model(X_train)

        # determine loss using predictions
        loss = loss_fn(pred, Y_train)

        # back-propagate and calculate gradient
        loss.backward()

        # update parameters using gradient
        optimizer.step()
        
        # display loss
        #logger.add_scalar('loss', loss, global_step=epoch)

        # predict using validation data
        pred = model(X_val)
    
        # determine accuracy
        acc = accuracy(pred, Y_val)

        print("loss: {}, accuracy: {}".format(loss, acc))
        #logger.add_scalar('accuracy', acc, global_step=epoch)
            
    save_model(model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model', choices=['linear', 'mlp'], default='linear')
    parser.add_argument('-r', '--rate', type=float, default=0.01)
    parser.add_argument('-e', '--epochs', type=int, default=10)
    # Put custom arguments here

    args = parser.parse_args()
    train(args)
