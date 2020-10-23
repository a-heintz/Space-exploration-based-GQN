"""
CircularOrbit_runGQN.py

Script to train the a GQN on the Circular-Orbit dataset
in accordance to the hyperparameter settings described in
the supplementary materials of the paper.
"""
import random
import math
from argparse import ArgumentParser

# Torch
import torch, os
import torch.nn as nn
from torch.distributions import Normal
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

# TensorboardX
from tensorboardX import SummaryWriter

# Ignite
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, Timer, Checkpoint
from ignite.metrics import RunningAverage

from gqn import GenerativeQueryNetwork, partition, Annealer
from CircularOrbit import CircularOrbit
from collections import OrderedDict
#from placeholder import PlaceholderData as ShepardMetzler

cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")
#print(cuda)
print('device: ', device)
# Random seeding
random.seed(99)
torch.manual_seed(99)
if cuda: torch.cuda.manual_seed(99)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    parser = ArgumentParser(description='Generative Query Network on Shepard Metzler Example')
    parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs run (default: 200)')
    parser.add_argument('--batch_size', type=int, default=1, help='multiple of batch size (default: 1)')
    parser.add_argument('--data_dir', type=str, help='location of data', default="train")
    parser.add_argument('--log_dir', type=str, help='location of logging', default="log")
    parser.add_argument('--fraction', type=float, help='how much of the data to uspip install e', default=1.0)
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--data_parallel', type=bool, help='whether to parallelise based on data (default: False)', default=False)
    parser.add_argument('--pretrained_path', type=str, help='location to save checkpoint models')
    parser.add_argument('--dataset_folder_length_train', type=int, default=900)
    parser.add_argument('--dataset_folder_length_test', type=int, default=100)
    parser.add_argument('--resume_training', type=str, default="False")
    args = parser.parse_args()
    print(args.resume_training)
    print('Creating GQN Model')
    # Create model and optimizer
    model = GenerativeQueryNetwork(x_dim=3, v_dim=7, r_dim=256, h_dim=128, z_dim=64, L=8).to(device)
    model = nn.DataParallel(model) if args.data_parallel else model
    
    #model = nn.DataParallel(model) if args.data_parallel else model

    optimizer = torch.optim.Adam(model.parameters(), lr=5 * 10 ** (-5))

    # Rate annealing schemes
    sigma_scheme = Annealer(2.0, 0.7, 80000)
    mu_scheme = Annealer(5 * 10 ** (-6), 5 * 10 ** (-6), 1.6 * 10 ** 5)
    print('Creating train dataset')
    # Load the dataset
    train_dataset = CircularOrbit(root_dir=args.data_dir, fraction=args.fraction)
    print('Creating test dataset')
    valid_dataset = CircularOrbit(root_dir=args.data_dir, fraction=args.fraction, train=False)

    kwargs = {'num_workers': args.workers, 'pin_memory': True} if cuda else {}
    print('train set:', len(train_dataset))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
    print('test set:', len(valid_dataset))
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
    #print(len(train_loader), len(valid_loader))
    def step(engine, batch):
        model.train()
        x, v = batch
        #print('CircOrbit_run-gqn.py: x shape', x.shape, ' -- v shape', v.shape)
        x, v = x.to(device), v.to(device)
        #x, v, x_q, v_q = partition(x, v)
        # Maximum number of context points to use
        _, b, m, *x_dims = x.shape
        _, b, m, *v_dims = v.shape

         # "Squeeze" the batch dimension
        images = x.view((-1, m, *x_dims))
        views = v.view((-1, m, *v_dims))
        
        # Sample random number of views
        n_context = 15
        indices = random.sample([i for i in range(m)], n_context)
    
        # Partition into context and query sets
        context_idx, query_idx = indices[:-1], indices[-1]

        x, v = images[:, context_idx], views[:, context_idx]
        x_q, v_q = images[:, query_idx], views[:, query_idx]

        # Reconstruction, representation and divergence
        x_mu, _, kl = model(x.float(), v.float(), x_q.float(), v_q.float())

        # Log likelihood
        sigma = next(sigma_scheme)
        ll = Normal(x_mu, sigma).log_prob(x_q)

        likelihood     = torch.mean(torch.sum(ll, dim=[1, 2, 3]))
        kl_divergence  = torch.mean(torch.sum(kl, dim=[1, 2, 3]))

        # Evidence lower bound
        elbo = likelihood - kl_divergence
        loss = -elbo
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        with torch.no_grad():
            # Anneal learning rate
            mu = next(mu_scheme)
            i = engine.state.iteration
            for group in optimizer.param_groups:
                group["lr"] = mu * math.sqrt(1 - 0.999 ** i) / (1 - 0.9 ** i)

        return {"elbo": elbo.item(), "kl": kl_divergence.item(), "sigma": sigma, "mu": mu}

    # Trainer and metrics
    trainer = Engine(step)
    metric_names = ["elbo", "kl", "sigma", "mu"]
    RunningAverage(output_transform=lambda x: x["elbo"]).attach(trainer, "elbo")
    RunningAverage(output_transform=lambda x: x["kl"]).attach(trainer, "kl")
    RunningAverage(output_transform=lambda x: x["sigma"]).attach(trainer, "sigma")
    RunningAverage(output_transform=lambda x: x["mu"]).attach(trainer, "mu")
    ProgressBar().attach(trainer, metric_names=metric_names)

    # Model checkpointing
    to_save = {'trainer': trainer, 'model': model, 'optimizer': optimizer, 'sigma_scheme': sigma_scheme, 'mu_scheme': mu_scheme}
    checkpoint_handler = ModelCheckpoint(args.pretrained_path, "checkpoint", n_saved=args.n_epochs, save_interval=1, require_empty=False)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler, to_save=to_save)#(every=1)
    
    #trainer.add_event_handler(event_name=Events.EPOCH_COMPLETED, handler=checkpoint_handler,
    #                          to_save={'model': model, 'optimizer': optimizer,
    #                                   'annealers': (sigma_scheme.data, mu_scheme.data)}) 

    timer = Timer(average=True).attach(trainer, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
                 pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)

    # Tensorbard writer
    writer = SummaryWriter(log_dir=args.log_dir)
    
    if args.resume_training == "True":
        checkpoints_dir = os.listdir(args.pretrained_path)
        #checkpoints_dir = [x for x in checkpoints_dir if 'checkpoint_checkpoint_' in x]
        #print(checkpoints_dir)
        resume_epoch = len(checkpoints_dir)
        
        checkpoint_path = os.path.join(args.pretrained_path, checkpoints_dir[-1])
        
        #to_load = {'model': model, 'optimizer': optimizer, 'sigma_scheme': sigma_scheme, 'mu_scheme': mu_scheme}
        to_load = to_save
        
        print(checkpoint_path)
        checkpoint = torch.load(checkpoint_path)
        #print(checkpoint)
        Checkpoint.load_objects(to_load=to_load, checkpoint=checkpoint)
        
        trainer.state.iteration = (resume_epoch) * len(train_loader)
        trainer.state.epoch = (resume_epoch)
        
        print('Resuming Training at Epoch ', trainer.state.epoch, '... Iteration ', trainer.state.iteration)

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_metrics(engine):
        for key, value in engine.state.metrics.items():
            writer.add_scalar("training/{}".format(key), value, engine.state.iteration)

    @trainer.on(Events.EPOCH_COMPLETED)
    def save_images(engine):
        with torch.no_grad():
            x, v = engine.state.batch
            x, v = x.to(device), v.to(device)
            x, v, x_q, v_q = partition(x, v)

            x_mu, r, _ = model(x, v, x_q, v_q)

            r = r.view(-1, 1, 16, 16)

            # Send to CPU
            x_mu = x_mu.detach().cpu().float()
            r = r.detach().cpu().float()

            writer.add_image("representation", make_grid(r), engine.state.epoch)
            writer.add_image("reconstruction", make_grid(x_mu), engine.state.epoch)

    @trainer.on(Events.EPOCH_COMPLETED)
    def validate(engine):
        model.eval()
        with torch.no_grad():
            x, v = next(iter(valid_loader))
            x, v = x.to(device), v.to(device)
            x, v, x_q, v_q = partition(x, v)

            # Reconstruction, representation and divergence
            x_mu, _, kl = model(x, v, x_q, v_q)

            # Validate at last sigma
            ll = Normal(x_mu, sigma_scheme.recent).log_prob(x_q)

            likelihood = torch.mean(torch.sum(ll, dim=[1, 2, 3]))
            kl_divergence = torch.mean(torch.sum(kl, dim=[1, 2, 3]))

            # Evidence lower bound
            elbo = likelihood - kl_divergence

            writer.add_scalar("validation/elbo", elbo.item(), engine.state.epoch)
            writer.add_scalar("validation/kl", kl_divergence.item(), engine.state.epoch)

    @trainer.on(Events.EXCEPTION_RAISED)
    def handle_exception(engine, e):
        writer.close()
        engine.terminate()
        if isinstance(e, KeyboardInterrupt) and (engine.state.iteration > 1):
            import warnings
            warnings.warn('KeyboardInterrupt caught. Exiting gracefully.')
            checkpoint_handler(engine, { 'model_exception': model })
        else: raise e

    print("Before")
    print(sigma_scheme.__repr__())
    print(mu_scheme.__repr__())
    trainer.run(train_loader, args.n_epochs)
    print("After")
    print(sigma_scheme.__repr__())
    print(mu_scheme.__repr__())
    writer.close()
