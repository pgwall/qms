import argparse, os, time, gc, copy

import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.metrics import mean_squared_error

import torch
import torch.utils.data as du
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import networkx as nx
import networkx.algorithms.components.connected as nxacc
import networkx.algorithms.dag as nxadag

from dataloaders import LoadFeatures
from dataloaders import LoadLayers

from qms_nn import qms_nn
import utils


def train_qms_nn(results_dir, train_file, val_file, lr, dropout, weight_decay, batch_size, cuda_id, ft, layers, save_hiddens, save_grads, save_model, keep_training_n, **kwargs):
    start_time = time.time()
    
    id2gene = ft.id2gene.copy()
    gene_dim = len(id2gene)
    
    feat_names = ft.feature_names
    model_input = ft.make_model_input()
    
    num_hiddens_gene = len(feat_names)
    
    model = qms_nn(num_hiddens_gene, dropout, id2gene, gene_dim, feat_names, layers,)
    
    optimizer = torch.optim.AdamW(model.parameters(), betas=(0.9, 0.99), eps=1e-05, lr=lr, weight_decay=weight_decay)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.2, eps=1e-10)
    
    num_params = utils.count_params(model)
    print(f'Number of model parameters: {num_params}')
    
    # Send model to GPU
    model.cuda(cuda_id)
    
    train_data = utils.load_drug_data(train_file, ft)
    val_data = utils.load_drug_data(val_file, ft)
    
    # Train args: train_data, val_data
    train_feats, train_labels = torch.Tensor(train_data[0]), torch.Tensor(train_data[1])
    val_feats, val_labels     = torch.Tensor(val_data[0]), torch.Tensor(val_data[1])
    
    # Train args: batch_size
    train_loader = du.DataLoader(du.TensorDataset(train_feats, train_labels), batch_size=batch_size, shuffle=True)
    val_loader   = du.DataLoader(du.TensorDataset(val_feats, val_labels), batch_size=batch_size, shuffle=False)
    
    mode = utils.set_mode(train_file, val_file)
    val_loss_list = []
    best_loss = 1e6
    epoch = 0
    
    training=True
    
    while training:
        epoch+=1
        epoch_start = time.time()

        model.train()
        train_predict, train_real_gpu, total_train_loss = run_batches('train', model, train_loader, model_input, save_hiddens, save_grads, results_dir, cuda_id, optimizer=optimizer)

        model.eval()
        val_predict, val_real_gpu, total_val_loss = run_batches(mode, model, val_loader, model_input, save_hiddens, save_grads, results_dir, cuda_id)

        scheduler.step(total_val_loss)

        train_corr = utils.pearson_corr(train_predict, train_real_gpu).item()
        val_corr   = utils.pearson_corr(val_predict, val_real_gpu).item()

        train_mse = mean_squared_error(train_real_gpu.cpu().numpy(), train_predict.cpu().numpy())
        val_mse   = mean_squared_error(val_real_gpu.cpu().numpy(), val_predict.cpu().numpy())
        
        val_loss_list.append(val_mse)

        epoch_time = time.time() - epoch_start
        print(f'epoch {epoch}\ttrain_corr {train_corr:.4f}\ttrain_mse {train_mse:.4f}\ttotal_train_loss {total_train_loss:.4f}\tval_corr {val_corr:.4f}\tval_mse {val_mse:.4f}\ttotal_val_loss {total_val_loss:.4f}\ttime {epoch_time:.1f}')

        if val_mse <= best_loss:
            best_loss = val_mse
            best_states = update_state_dicts(model, optimizer, save_model,)
            best_results = {'best_epoch_mse':val_mse,
                           'best_epoch':epoch, 
                           'best_epoch_tr_mse':train_mse,     
                           'best_epoch_tr_corr':train_corr}
            best_preds = val_predict.cpu().numpy()

        training = keep_training(val_loss_list, epoch, keep_training_n)
        #training = False

        if not training:
            train_time = time.time() - start_time
            best_results['n_epochs'] = epoch
            best_results['train_time'] = train_time
            best_results['val_loss_list'] = val_loss_list
            
            max_mem = torch.cuda.max_memory_allocated(device=cuda_id)/(1024**3)
            print(f'\nMax GPU memory occupied during RUN: {max_mem:.3f}GB\n')
            
            print(f'Training complete. Time to train: {train_time/60:.2f} minutes')
        
        utils.save_predictions(mode, val_data, best_preds, results_dir)
            
    return best_results, best_states




def run_batches(mode, model, data_loader, model_input, save_hiddens, save_grads, results_dir, cuda_id, optimizer=None):
    """
    Run batches for one epoch, save preds/grads/hiddens as needed
    """

    epoch_predict = torch.zeros(0, 0).type(torch.FloatTensor).cuda(cuda_id)
    epoch_real = torch.zeros(0, 0).type(torch.FloatTensor).cuda(cuda_id)
    total_epoch_loss = 0

    for i, (input_data, input_labels) in enumerate(data_loader):
        features, labels = utils.build_input_features(model_input, input_data, input_labels)

        cuda_features = torch.Tensor(features).cuda(cuda_id)
        cuda_labels   = torch.Tensor(labels).cuda(cuda_id)

        aux_out_map, out_map, out_map_gene, input_map = model(cuda_features)

        if epoch_predict.size()[0] == 0:
            epoch_predict = aux_out_map['final'].data
            epoch_real    = cuda_labels
        else:
            epoch_predict = torch.cat([epoch_predict, aux_out_map['final'].data], dim=0)
            epoch_real    = torch.cat([epoch_real, cuda_labels], dim=0)

        batch_loss = 0
        for name, output in aux_out_map.items():
            loss = nn.MSELoss()
            if name == 'final':
                batch_loss += loss(output, cuda_labels)
            else:  
                batch_loss += 0.2 * loss(output, cuda_labels)

        total_epoch_loss += batch_loss.item()/features.shape[0]

        if mode == 'train':
            batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        if mode == 'test' and save_hiddens:
            utils.save_hiddens([out_map, out_map_gene], results_dir)

        if mode == 'test' and save_grads:
            utils.save_gradients(aux_out_map, [out_map, out_map_gene, input_map], results_dir)

        #Memory cleanup
        del features
        del cuda_features
        del cuda_labels
        gc.collect()
        torch.cuda.empty_cache()
    
    return epoch_predict, epoch_real, total_epoch_loss


def keep_training(loss_list, current_epoch, keep_training_n=20, keep_training_delta=1e-4, max_epochs=50):
    """
    Function controls stoppage for main_train() function. 

    Checks to see if the best loss was in the last n epochs. If yes, then keep training. If best loss was > n epochs ago, then stop.
    Also checks to see if the best loss happened in the first <keep_training_n> epochs. If so, then keep training until a better loss is achieved in case this is just a random result. 
    """
    # Ignore early epochs
    length_ok = len(loss_list) >=  keep_training_n

    # Stop if epoch limit reached 
    if current_epoch == max_epochs:
        training = False

    # Keep training if model still learns with decent performance improvement 
    elif length_ok:
        min_previous_losses = min(loss_list[:-keep_training_n+1])
        last_n_min = min(loss_list[-keep_training_n+1:])

        delta = abs(min_previous_losses - last_n_min)

        still_learning  = last_n_min <= min_previous_losses
        still_improving = delta >= keep_training_delta

        training = still_learning and still_improving

    # Stop learning if no improvement
    else:
        training = True

    return training


def update_state_dicts(model, optimizer, save_model,):
    states = {}
    if save_model:
        states['model_state_dict'] = copy.deepcopy(model.state_dict())
        states['optimizer_state_dict'] = copy.deepcopy(optimizer.state_dict())
    return states


def save_state_dicts(best_states, results_dir):
    model_path = os.path.join(results_dir, 'model.pt')
    optim_path = os.path.join(results_dir, 'optimizer.pt')
        
    if len(best_states) > 0:
        torch.save(best_states['model_state_dict'], model_path)
        torch.save(best_states['optimizer_state_dict'], optim_path)


# Define parameters ##################################
parser = argparse.ArgumentParser(description = 'Train QMS drug response model')
parser.add_argument('-model_yml', help='Filepath to model settings YAML file', default='')
opt = parser.parse_args()

settings = utils.read_yml(opt.model_yml)

ft = LoadFeatures(settings['features'])
layers = LoadLayers()

# Create the directory to store results and model
results_dir = utils.create_results_directory(settings)



# Train model ##################################

best_results, best_states = train_qms_nn(ft=ft, layers=layers, **settings)

################################################



save_state_dicts(best_states, results_dir)

utils.save_results(settings, best_results, results_dir)

utils.write_yml(os.path.join(results_dir, 'model_settings.yml'), settings)

# TODO:
# * Update keep_training 