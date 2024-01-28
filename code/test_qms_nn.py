import argparse, os, time, gc, copy

import numpy as np
import pandas as pd

import torch
import torch.utils.data as du
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import scipy.stats as stats
from sklearn.metrics import mean_squared_error

from dataloaders import LoadFeatures
from dataloaders import LoadOntology

from qms_nn import qms_nn
from utils import *



def test_qms_nn(model_dir, saved_model_dir, test_file, num_hiddens_gene, lr, dropout, weight_decay, batch_size, cuda_id, save_hiddens, save_grads, features, ont, ):
    start_time = time.time()
    
    id2gene = features.id2gene.copy()
    gene_dim = len(id2gene)
    
    feat_names = features.feature_names
    model_input = features.make_model_input()
    
    model = load_saved_model(qms_nn, saved_model_dir, num_hiddens_gene, dropout, id2gene, gene_dim, feat_names, ont,)
    
    num_params = count_params(model)
    print(f'Number of model parameters: {num_params}')
    
    # Send model to GPU
    model.cuda(cuda_id)
    
    test_data = load_drug_data(test_file, features)
    
    test_feats, test_labels = torch.Tensor(test_data[0]), torch.Tensor(test_data[1])
    
    test_loader = du.DataLoader(du.TensorDataset(test_feats, test_labels), batch_size=batch_size, shuffle=False)

    model.eval()
    test_predict, test_real_gpu, total_test_loss = run_batches('test', model, test_loader, model_input, save_hiddens, save_grads)

    test_corr = pearson_corr(test_predict, test_real_gpu).item()
    test_mse = mean_squared_error(test_real_gpu.cpu().numpy(), test_predict.cpu().numpy())

    save_predictions('test', test_data, test_predict.cpu().numpy(), model_dir)
    
    test_results = {'test_corr':test_corr, 'test_mse':test_mse}
            
    return test_results


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



parser = argparse.ArgumentParser(description = 'Test QMS drug response model')

parser.add_argument('-saved_model_dir', help = 'Directory with saved model to test', default='')
parser.add_argument('-model_dir', help = 'Specify name /qms/models/< model_dir_name >; defaults to name of train/test file', default='')
parser.add_argument('-features', help = 'Specify features for training the model', default=[])
parser.add_argument('-cells', help = 'Cell line names for features', type = list, default=[])
parser.add_argument('-genes', help = 'Gene names names for features', default=[])
parser.add_argument('-train_file', help='Training dataset', type=str, default='')
parser.add_argument('-val_file', help='Training dataset', type=str, default='')
parser.add_argument('-test_file', help='Training dataset', type=str, default='')
parser.add_argument('-save_model', help='Save model', type=bool, default=False)
parser.add_argument('-save_grads', help='Save gradients of input features and gene embeddings', type=bool, default=False)
parser.add_argument('-cuda_id', help = 'Specify GPU', type = int, default = 0)

parser.parse_args()

# Load results and model info from trained model #####################
parser = argparse.ArgumentParser(description = 'Test QMS drug response model')
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

saved_model_dir = settings['model_dir']

model_name = settings['model_folder_name']
if not model_name:
    model_name = test_file.split('.')[0]
model_dir = os.path.abspath('../models/'+model_name)
os.makedirs(model_dir, exist_ok=True)

model_dir, model_name = create_model_directory(settings['model_folder_name'])

settings['model_dir'] = model_dir
settings['model_name'] = model_name
settings['num_hiddens_gene'] = num_hiddens_gene

test_results = test_qms_nn(model_dir, saved_model_dir, test_file, num_hiddens_gene, lr, dropout, 
                           weight_decay, batch_size, cuda_id, save_hiddens, save_grads, features, ont)

save_results(model_info, test_results, model_dir)

write_yml(os.path.join(model_dir, 'test_model_settings.yml'), settings)