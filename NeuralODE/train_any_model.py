import torch
import numpy as np
from model_factory import ModelFactory
from dataset import TrajectoryDataset
from systems import pendulum_time_invariant
import random
import os
from omegaconf import OmegaConf
from fire import Fire
from tqdm import tqdm
from utils import create_learning_gif

# TODO
# - Get it to work for RNN TOO
# - add noise to samples

def seed_all(seed):

    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train(exp_dir, system, system_cfg, model_names, model_cfg, train_cfg, model_name):

    seed_all(train_cfg.seed) # Make sure different Model initializations still use the same seed (for display data)

    T,B = system_cfg[system].n_sample_steps, train_cfg.batch_size

    if system == 'pendulum':
        n_dim = 2
        ode = pendulum_time_invariant
    elif system == 'lorenz':
        n_dim = 3
        ode = ...

    factory = ModelFactory()
    model = factory.get_model(model_name=model_name, d_in=n_dim, d_out=n_dim, **model_cfg[model_name])

    dataset = TrajectoryDataset(
        ode=ode, 
        samples_per_epoch=train_cfg.training_iters, 
        num_timesteps_out=T, 
        dt_solver=system_cfg[system].dt_solver, 
        dt_out=system_cfg[system].dt_out
        )
    
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=train_cfg.batch_size)

    if model_cfg[model_name]['optimizer'] == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=model_cfg[model_name]['learning_rate'])
    elif model_cfg[model_name]['optimizer'] == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=model_cfg[model_name]['learning_rate'])
    loss_fn = torch.nn.MSELoss()

    last_loss = 1e6
    running_loss = 0.0

    result_gt, time_gt = next(iter(dataloader)) # (B,T,n_dim), (B,T)
    result_model = []

    with tqdm(total=len(dataloader)) as pbar:

        for i, (states,t) in enumerate(dataloader):

            optimizer.zero_grad()
            pbar.set_description(f'Training model {model_name}, iter: {i} / {len(dataloader)}, last loss: {last_loss:.5f}.')

            states = states.detach(); states.requires_grad_(True)
            t = t.detach(); t.requires_grad_(True)

            def get_subset(x, t, T_subset):
                B, T, D = x.shape
                start_indices = torch.randint(0, T_subset, (B,))
                indices = torch.arange(T_subset).unsqueeze(0) + start_indices.unsqueeze(1)
                return states[torch.arange(B).unsqueeze(1), indices], t[torch.arange(B).unsqueeze(1), indices]
            
            states_subset, t_subset = get_subset(states, t, train_cfg.points_per_sample)

            pred = model(*factory.get_model_input(model_name, states_subset, t_subset))
            loss = loss_fn( factory.get_model_output(model_name,pred), states_subset)
            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            if i%train_cfg.log_every == train_cfg.log_every-1:

                last_loss = running_loss/train_cfg.log_every
                with torch.no_grad():

                    pred = model(*factory.get_model_input(model_name, result_gt, time_gt))
                    result_model.append(factory.get_model_output(model_name, pred))

                running_loss=0.0

            pbar.update()
    
    model_weights_fn = os.path.join(exp_dir, model_name + '.pt')
    torch.save(factory.get_model_statedict(model_name, model), model_weights_fn)
    print(f'Saved model and settings to {exp_dir}')

    return result_gt, result_model


def main(config=None, **kwargs):
    
    config = OmegaConf.load(config) if (config is not None) else {}
    assert config, (f'No config provided, use python NeuralODE/train.py --config=NeuralODE/config.yml,' 
                    'or other vscode launch configuration')
    run = 0
    basename = os.path.join('NeuralODE', 'eval', f'run_')
    while os.path.exists(basename + str(run).zfill(2)):
        run +=1
    config.exp_dir = basename + str(run).zfill(2)
    os.makedirs(config.exp_dir, exist_ok=True)
    OmegaConf.save(config, os.path.join(config.exp_dir, 'config.yml'))

    print('\nConfiguration:         \n--------------------\n'
         f'{OmegaConf.to_yaml(config)}--------------------\n')

    results = {}
    for mname in config.model_names:
        results_gt, results_model = train(**config, model_name=mname)
        results['ground_truth'] = results_gt
        results[mname] = results_model

    create_learning_gif(results, config.model_names, config.train_cfg.log_every, exp_dir=config.exp_dir)


if __name__ == "__main__":
    Fire(main)



