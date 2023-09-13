
import torch.nn as nn
import torch
import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal
import os
from partial_bnn_functional import *
from PartialNLP.uci import UCIDataloader, UCIDataset
from PartialNLP.MAP_baseline.MapNN import MapNN
def calculate_nll_vi(model, dataloader, sigma, num_mc_runs = 600, device = 'cpu'):

    mc_overall = []
    for batch, label in dataloader:
        mc_batch = []
        batch = batch.to(device)
        label = label.to(device)
        for mc_run in range(num_mc_runs):
            output = model(batch)
            dist = MultivariateNormal(output, torch.eye(output.shape[0])*sigma)
            mc_batch.append(dist.log_prob(label).item())

        mc_overall.append(np.mean(mc_batch))

    nll = np.mean(mc_overall)

    return nll

def calculate_mse(model, dataloader, num_mc_runs = 600, device = 'cpu'):
    mc_overall = []
    for batch, label in dataloader:
        mc_batch = []
        batch = batch.to(device)
        label = label.to(device)
        for mc_run in range(num_mc_runs):
            output = model(batch)
            mc_batch.append((label - output)**2)

        mc_overall.append(torch.mean(mc_batch).item())

    mse = np.mean(mc_overall)

    return mse

def calculate_sigma(predictions, labels):
    residuals = predictions - labels
    return np.var(residuals)

def make_multiple_runs_vi(dataset_class, data_path, model_path, num_runs, device='cpu', gap=True, train_args = None)
    for run in range(num_runs):
        dataset = dataset_class(data_dir=data_path,
                                test_split_type="gap" if gap else "random",
                                test_size=0.1,
                                gap_column='random',
                                val_fraction_of_train=0.1,
                                seed=np.random.randint(0, 1000))

        n_train, p = dataset.X_train.shape
        n_val = dataset.X_val.shape[0]
        out_dim = dataset.y_train.shape[1]
        n_test = dataset.X_test.shape[0]

        if device == 'cuda' and not torch.cuda.is_available():
            device = 'cpu'


        percentages = [1,2,5, 8, 14, 23, 37, 61, 100]
        results = {
            'percentages': percentages,
            'x_train': dataset.X_train,
            'y_train': dataset.y_train,
            'y_val': dataset.y_val,
            'x_val': dataset.X_val,
            'x_test': dataset.X_test,
            'y_test': dataset.y_test,


        }
        train_dataloader = DataLoader(UCIDataloader(dataset.X_train, dataset.y_train), batch_size=n_train)
        val_dataloader = DataLoader(UCIDataloader(dataset.X_val, dataset.y_val), batch_size=n_val)
        test_dataloader = DataLoader(UCIDataloader(dataset.X_test, dataset.y_test), batch_size=n_test)

        # train_partial_with_accumulated_stochasticity(MapNN(p, 50, 2, out_dim, "leaky_relu"),
        #                                      train_dataloader,val_dataloader,test_dataloader, train_args)

        model, res = train_model_with_varying_stochasticity_scheme_two(MapNN(p, 35, 2, out_dim, "leaky_relu"),
                                                                  train_dataloader,
                                                                  val_dataloader,
                                                                  percentages,
                                                                  train_args,
                                                                  run,
                                                                  dataloader_test=test_dataloader
                                                                  )

        train_res, val_res, test_res = res

        sigma = calculate_sigma(*train_res)

        results['val_nll'] = calculate_nll_vi(model, val_dataloader, sigma, train_args['num_mc_runs'], device)
        results['val_mse'] = calculate_mse(model, val_dataloader,train_args['num_mc_runs'], device)

        results['test_nll'] = calculate_nll_vi(model, val_dataloader, sigma, train_args['num_mc_runs'], device)
        results['test_mse'] = calculate_mse(model, val_dataloader,train_args['num_mc_runs'], device)

