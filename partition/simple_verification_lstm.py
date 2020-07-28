import argparse
import random
import pickle
import os
import pdb
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from partition.lstm import LSTM
from partition.data_utils import load_data, get_batches
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
from auto_LiRPA.utils import AverageMeter, logger
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import *

## Step 1: Define computational graph by implementing forward()
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
parser.add_argument("--norm", type=int, default=2)
parser.add_argument("--eps", type=float, default=0.1)
parser.add_argument("--num_epochs", type=int, default=20)  
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--num_slices", type=int, default=8)
parser.add_argument("--hidden_size", type=int, default=64)
parser.add_argument("--num_classes", type=int, default=10) 
parser.add_argument("--input_size", type=int, default=784)
parser.add_argument("--lr", type=float, default=5e-3)
parser.add_argument("--dir", type=str, default="model", help="directory to load or save the model")
parser.add_argument("--num_epochs_warmup", type=int, default=1, help="number of epochs for the warmup stage when eps is linearly increased from 0 to the full value")
parser.add_argument("--log_interval", type=int, default=10, help="interval of printing the log during training")
args = parser.parse_args()   


def step(model, ptb, batch, eps=args.eps, train=False):
    ptb.set_eps(eps)    
    X, y = model.get_input(batch)
    X = BoundedTensor(X, ptb)
    logits = model.core(X)

    num_class = args.num_classes
    c = torch.eye(num_class).type_as(X)[y].unsqueeze(1) - \
        torch.eye(num_class).type_as(X).unsqueeze(0)
    I = (~(y.data.unsqueeze(1) == torch.arange(num_class).type_as(y.data).unsqueeze(0)))
    c = (c[I].view(X.size(0), num_class - 1, num_class))

    lb, ub = model.core.compute_bounds(IBP=True, C=c, method='backward', bound_upper=False)

    lb_padded = torch.cat((torch.zeros(size=(lb.size(0),1), dtype=lb.dtype, device=lb.device), lb), dim=1)
    fake_labels = torch.zeros(size=(lb.size(0),), dtype=torch.int64, device=lb.device)
    acc = (torch.argmax(logits, dim=-1) == y).float().mean()
    acc_robust = 1 - torch.mean((lb < 0).any(dim=1).float())    
    loss = nn.CrossEntropyLoss()(-lb_padded, fake_labels)

    if train:
        loss.backward()

    return acc.detach(), acc_robust.detach(), loss.detach()

'''
## Step 2: Prepare dataset as usual
test_data = torchvision.datasets.MNIST("./data", train=False, download=True, transform=torchvision.transforms.ToTensor())
# For illustration we only use 2 image from dataset
N = 2
n_classes = 10
image = test_data.data[:N].view(N,1,28,28).cuda()
# Convert to float
image = image.to(torch.float32) / 255.0

## Step 3: wrap model with auto_LiRPA
# The second parameter is for constructing the trace of the computational graph, and its content is not important.
model = BoundedModule(model, torch.empty_like(image), device="cuda")

## Step 4: Compute bounds using LiRPA given a perturbation
eps = 0.3
norm = np.inf
ptb = PerturbationLpNorm(norm=norm, eps=eps)
image = BoundedTensor(image, ptb)
'''



data_train, data_test = load_data()
logger.info("Dataset sizes: {}/{}".format(len(data_train), len(data_test)))

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

model = LSTM(args).to(args.device)  
checkpoint = torch.load(os.path.join(os.path.dirname(__file__),"model/model"))
# Load the pretrained weights (from vision)
#checkpoint = torch.load(os.path.join(os.path.dirname(__file__),"pretrain/mnist_cnn_small.pth"))
model.load_state_dict(checkpoint)

test_batches = get_batches(data_test, args.batch_size) 
X, y = model.get_input(test_batches[0])
ptb = PerturbationLpNorm(norm=args.norm, eps=args.eps) 
X = BoundedTensor(X, ptb)
model.core = BoundedModule(model.core, (X,), device=args.device)
optimizer = model.build_optimizer()

avg_acc, avg_acc_robust, avg_loss = avg = [AverageMeter() for i in range(3)]

# Load the pretrained weights
#checkpoint = torch.load(os.path.join(os.path.dirname(__file__),"./model"))
#print(checkpoint)
#model.load_state_dict(checkpoint)

def infer(epoch, batches, type):
    model.eval()
    for a in avg: a.reset()    
    for i, batch in enumerate(batches):
        acc, acc_robust, loss = res = step(model, ptb, batch)
        for k in range(3):
            avg[k].update(res[k], len(batch))                 
    logger.info("Epoch {}, {}: acc {:.3f}, acc_robust {:.3f}, loss {:.5f}".format(
        epoch, type, avg_acc.avg, avg_acc_robust.avg, avg_loss.avg))

def main():
    for t in range(model.checkpoint, args.num_epochs):
        infer(t + 1, test_batches, "test")
    if model.checkpoint == args.num_epochs:
        infer(args.num_epochs, test_batches, "test")

if __name__ == "__main__":
    main()
# Get model prediction as usual
#pred = model(image)
#label = torch.argmax(pred, dim=1).cpu().numpy()
# Compute bounds
#lb, ub = model.compute_bounds()

## Step 5: Final output
#pred = pred.detach().cpu().numpy()
'''
lb = lb.detach().cpu().numpy()
ub = ub.detach().cpu().numpy()
for i in range(N):
    print("Image {} top-1 prediction {}".format(i, label[i]))
    for j in range(n_classes):
        print("f_{j}(x_0) = {fx0:8.3f},   {l:8.3f} <= f_{j}(x_0+delta) <= {u:8.3f}".format(j=j, fx0=pred[i][j], l=lb[i][j], u=ub[i][j]))
    print()
'''
