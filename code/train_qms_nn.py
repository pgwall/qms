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
from dataloaders import LoadOntology

from qms_nn import qms_nn
from utils import *


def train_qms_nn(model_dir, train_file, val_file, num_hiddens_gene, lr, dropout, weight_decay, batch_size, cuda_id, features, ont, save_hiddens, save_grads):
    start_time = time.time()
    
    id2gene = features.id2gene.copy()
    gene_dim = len(id2gene)
    
    feat_names = features.feature_names
    model_input = features.make_model_input()
    
    # 
    model = qms_nn(num_hiddens_gene, dropout, id2gene, gene_dim, feat_names, ont,)
    optimizer = torch.optim.AdamW(model.parameters(), betas=(0.9, 0.99), eps=1e-05, lr=lr, weight_decay=weight_decay)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.2, eps=1e-10)
    
    num_params = count_params(model)
    print(f'Number of model parameters: {num_params}')
    
    # Send model to GPU
    model.cuda(cuda_id)
    
    train_data = load_drug_data(train_file, features)
    val_data = load_drug_data(val_file, features)
    
    # Train args: train_data, val_data
    train_feats, train_labels = torch.Tensor(train_data[0]), torch.Tensor(train_data[1])
    val_feats, val_labels     = torch.Tensor(val_data[0]), torch.Tensor(val_data[1])
    
    # Train args: batch_size
    train_loader = du.DataLoader(du.TensorDataset(train_feats, train_labels), batch_size=batch_size, shuffle=True)
    val_loader   = du.DataLoader(du.TensorDataset(val_feats, val_labels), batch_size=batch_size, shuffle=False)
    
    mode = set_mode(train_file, val_file)
    val_loss_list = []
    best_loss = 1e6
    epoch = 0
    
    training=True
    
    while training:
        epoch+=1
        epoch_start = time.time()

        model.train()
        train_predict, train_real_gpu, total_train_loss = run_batches('train', model, train_loader, model_input, save_hiddens, save_grads, optimizer=optimizer)

        model.eval()
        val_predict, val_real_gpu, total_val_loss = run_batches(mode, model, val_loader, model_input, save_hiddens, save_grads)

        scheduler.step(total_val_loss)

        train_corr = pearson_corr(train_predict, train_real_gpu).item()
        val_corr   = pearson_corr(val_predict, val_real_gpu).item()

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

        #training = keep_training(val_loss_list, epoch)
        training = False

        if not training:
            train_time = time.time() - start_time
            best_results['n_epochs'] = epoch
            best_results['train_time'] = train_time
            best_results['val_loss_list'] = val_loss_list
            
            print(f'Training complete. Time to train: {train_time/60:.2f} minutes')
        
        save_predictions(mode, val_data, best_preds, model_dir)
            
    return best_results, best_states




def run_batches(mode, model, data_loader, model_input, save_hiddens, save_grads, optimizer=None):
    """
    Run epoch batches for mode = train, val, or test
    """

    epoch_predict = torch.zeros(0, 0).type(torch.FloatTensor).cuda(cuda_id)
    epoch_real = torch.zeros(0, 0).type(torch.FloatTensor).cuda(cuda_id)
    total_epoch_loss = 0

    for i, (input_data, input_labels) in enumerate(data_loader):
        features, labels = build_input_features(model_input, input_data, input_labels)

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
            save_hiddens([out_map, out_map_gene], model_dir)

        if mode == 'test' and save_grads:
            save_gradients(aux_out_map, [out_map, out_map_gene, input_map], model_dir)

        #Memory cleanup
        del features
        del cuda_features
        del cuda_labels
        gc.collect()
        torch.cuda.empty_cache()
    # Both are still on GPU
    return epoch_predict, epoch_real, total_epoch_loss


def keep_training(loss_list, current_epoch, keep_training_n=10, keep_training_delta=1e-4, max_epochs=50):
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


def save_state_dicts(best_states, model_dir):
    model_path = os.path.join(model_dir, 'model.pt')
    optim_path = os.path.join(model_dir, 'optimizer.pt')
        
    if len(best_states) > 0:
        torch.save(best_states['model_state_dict'], model_path)
        torch.save(best_states['optimizer_state_dict'], optim_path)


# Define parameters ##################################
parser = argparse.ArgumentParser(description = 'Train QMS drug response model')
parser.add_argument('-model_yml', help='Filepath to model configuration YAML file', default='')
opt = parser.parse_args()

settings = read_yml(opt.model_yml)

train_file = settings['train_file']
val_file = settings['val_file']
test_file = settings['test_file']

save_model = settings['save_model']
save_grads = settings['save_grads']

features = LoadFeatures(settings['features'])
ont = LoadOntology()

num_hiddens_gene = len(features.feature_names)
weight_decay = settings['weight_decay']
batch_size = settings['batch_size']
dropout = settings['dropout']
lr = settings['lr']
cuda_id = settings['cuda_id']

model_name = settings['model_folder_name']
if not model_name:
    model_name = train_file.split('.')[0]
model_dir = os.path.abspath('../models/'+model_name)
os.makedirs(model_dir, exist_ok=True)

settings['model_dir'] = model_dir
settings['model_name'] = model_name
settings['num_hiddens_gene'] = num_hiddens_gene



# Train model ##################################
best_results, best_states = train_qms_nn(model_dir, train_file, val_file, num_hiddens_gene, lr, dropout, 
                                         weight_decay, batch_size, cuda_id, features, ont, save_hiddens, save_grads )

save_state_dicts(best_states, model_dir)

save_results(settings, best_results, model_dir)

write_yml(os.path.join(model_dir, 'model_settings.yml'), settings)