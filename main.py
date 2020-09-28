import argparse, torch, random
import numpy as np
from torchvision import datasets, transforms
import torch.nn.functional as F
from models import SNN, SpikeCELoss

parser = argparse.ArgumentParser(description='Training a SNN on MNIST with EventProp')

# General settings
parser.add_argument('--data-folder', type=str, default='data', help='name of folder to place dataset (default: data)')
parser.add_argument('--device', type=str, default='cuda', help='device to run on (default: cuda)')
parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')
parser.add_argument('--print-freq', type=int, default=100, help='training stats are printed every so many batches (default: 100)')
parser.add_argument('--deterministic', action='store_true', help='run in deterministic mode for reproducibility')

# Training settings
parser.add_argument('--epochs', type=int, default=40, help='number of epochs to train (default: 100)')
parser.add_argument('--lr', type=float, default=1.0, help='learning rate (default: 1.0)')
parser.add_argument('--batch-size', type=int, default=128, help='size of batch used for each update step (default: 128)')

# Loss settings (specific for SNNs)
parser.add_argument('--xi', type=float, default=0.4, help='constant factor for cross-entropy loss (default: 0.4)')
parser.add_argument('--alpha', type=float, default=0.01, help='regularization factor for early-spiking (default: 0.01)')
parser.add_argument('--beta', type=float, default=2, help='constant factor for regularization term (default: 2.0)')

# Spiking Model settings
parser.add_argument('--T', type=float, default=20, help='duration for each simulation, in ms (default: 20)')
parser.add_argument('--dt', type=float, default=1, help='time step to discretize the simulation, in ms (default: 1)')
parser.add_argument('--tau_m', type=float, default=20.0, help='membrane time constant, in ms (default: 20)')
parser.add_argument('--tau_s', type=float, default=5.0, help='synaptic time constant, in ms (default: 5)')
parser.add_argument('--t_max', type=float, default=12.0, help='max input spiking time, in ms (default: 12)')
parser.add_argument('--t_min', type=float, default=2.0, help='min input spiking time, in ms (default: 2)')

args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

if args.deterministic:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def encode_data(data):
    spike_data = args.t_min + (args.t_max - args.t_min) * (data < 0.5).view(data.shape[0], -1)
    spike_data = F.one_hot(spike_data.long(), int(args.T))
    return spike_data

def train(model, criterion, optimizer, loader):
    total_correct = 0.
    total_loss = 0.
    total_samples = 0.
    model.train()
    
    for batch_idx, (input, target) in enumerate(loader):
        input, target = input.to(args.device), target.to(args.device)
        input = encode_data(input)
        
        total_correct = 0.
        total_loss = 0.
        total_samples = 0.
        
        output = model(input)

        loss = criterion(output, target)

        if args.alpha != 0:
            target_first_spike_times = output.gather(1, target.view(-1, 1))
            loss += args.alpha * (torch.exp(target_first_spike_times / (args.beta * args.tau_s)) - 1).mean()

        predictions = output.data.min(1, keepdim=True)[1]
        total_correct += predictions.eq(target.data.view_as(predictions)).sum().item()
        total_loss += loss.item() * len(target)
        total_samples += len(target)
        
        optimizer.zero_grad()
        loss.backward()
        
        optimizer.step()

        if batch_idx % args.print_freq == 0:
            print('\tBatch {:03d}/{:03d}: \tAcc {:.2f}  Loss {:.3f}'.format(batch_idx, len(loader), 100*total_correct/total_samples, total_loss/total_samples))
   
    print('\t\tTrain: \tAcc {:.2f}  Loss {:.3f}'.format(100*total_correct/total_samples, total_loss/total_samples))

def test(model, loader):
    total_correct = 0.
    total_samples = 0.
    model.eval()
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):
            data, target = data.to(args.device), target.to(args.device)
            spike_data = encode_data(data)
            
            first_post_spikes = model(spike_data)
            predictions = first_post_spikes.data.min(1, keepdim=True)[1]
            total_correct += predictions.eq(target.data.view_as(predictions)).sum().item()
            total_samples += len(target)
            
        print('\t\tTest: \tAcc {:.2f}'.format(100*total_correct/total_samples))

train_dataset = datasets.MNIST(args.data_folder, train=True, download=True, transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_dataset = datasets.MNIST(args.data_folder, train=False, download=True, transform=transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        
model = SNN(784, 10, args.T, args.dt, args.tau_m, args.tau_s).to(args.device)
criterion = SpikeCELoss(args.T, args.xi, args.tau_s)
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

for epoch in range(args.epochs):
    print('Epoch {:03d}/{:03d}'.format(epoch, args.epochs))
    train(model, criterion, optimizer, train_loader)
    test(model, test_loader)
    scheduler.step()
