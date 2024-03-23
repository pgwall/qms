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
from dataloaders import LoadLayers

from qms_nn import qms_nn
import utils



def test_qms_nn(model_path, results_dir, test_file, batch_size, cuda_id, save_hiddens, save_grads, ft, layers, **kwargs):
    start_time = time.time()
    
    # Load gene index map in gene2ind.txt file
    id2gene = ft.id2gene.copy()
    gene_dim = len(id2gene)
    
    # Load names of input features --> used to store input feature gradients/hidden activations
    feat_names = ft.feature_names
    input_features = ft.make_model_input()
    
    num_hiddens_gene = len(feat_names)
    
    # Load pre-trained model in eval() mode
    model = utils.load_saved_model(qms_nn, model_path, num_hiddens_gene, id2gene, gene_dim, feat_names, layers,)
    
    num_params = utils.count_params(model)
    print(f'Number of model parameters: {num_params}')
    
    model.cuda(cuda_id)
    
    test_data = utils.load_drug_data(test_file, ft)
    
    test_feats, test_labels = torch.Tensor(test_data[0]), torch.Tensor(test_data[1])
    
    test_loader = du.DataLoader(du.TensorDataset(test_feats, test_labels), batch_size=batch_size, shuffle=False)

    model.eval()
    
    # Run model
    test_predict, test_real_gpu = run_batches(model, test_loader, input_features, save_hiddens, save_grads, results_dir, cuda_id)

    test_corr = utils.pearson_corr(test_predict, test_real_gpu).item()
    test_mse = mean_squared_error(test_real_gpu.cpu().numpy(), test_predict.cpu().numpy())

    utils.save_predictions('test', test_data, test_predict.cpu().numpy(), results_dir)
    
    test_results = {'test_corr':test_corr, 'test_mse':test_mse}
            
    return test_results


def run_batches(model, test_loader, input_features, save_hiddens, save_grads, results_dir, cuda_id):
    """
    Method for running test set batches
    """
    
    # Save real drug response and predicted drug response
    epoch_predict = torch.zeros(0, 0).type(torch.FloatTensor).cuda(cuda_id)
    epoch_real = torch.zeros(0, 0).type(torch.FloatTensor).cuda(cuda_id)

    # Iterate through test data
    for i, (input_data, input_labels) in enumerate(test_loader):
        features, labels = utils.build_input_features(input_features, input_data, input_labels)

        cuda_features = torch.Tensor(features).cuda(cuda_id)
        cuda_labels   = torch.Tensor(labels).cuda(cuda_id)
        
        # predictions  --> predicted AUDRC for test samples
        # out_map_gene --> hidden activation states of gene layer
        # input_map    --> input features to each gene layer
        aux_out_map, _, out_map_gene, input_map = model(cuda_features)

        if epoch_predict.size()[0] == 0:
            epoch_predict = aux_out_map['final'].data
            epoch_real    = cuda_labels
        else:
            epoch_predict = torch.cat([epoch_predict, aux_out_map['final'].data], dim=0)
            epoch_real    = torch.cat([epoch_real, cuda_labels], dim=0)
        
        # Save hidden activation states
        if save_hiddens:
            utils.save_hiddens([out_map_gene], results_dir)
        
        # Save gradients of model features
        if save_grads:
            utils.save_gradients(aux_out_map, [out_map_gene, input_map], results_dir)

        #Memory cleanup
        del features
        del cuda_features
        del cuda_labels
        gc.collect()
        torch.cuda.empty_cache()

    return epoch_predict, epoch_real


# Run pre-trained model ################################################################
parser = argparse.ArgumentParser(description = 'Test QMS drug response model')
parser.add_argument('-model_yml', help='Filepath to model configuration YAML file', default='')
opt = parser.parse_args()

# Load model settings 
settings = utils.read_yml(opt.model_yml)

ft = LoadFeatures(settings['features'])
layers = LoadLayers()

# Create the directory to store results
results_dir = utils.create_results_directory(settings)


# Run model ##########################################################################

test_results = test_qms_nn(ft=ft, layers=layers, **settings)

######################################################################################


# Save results
utils.save_results(settings, test_results, results_dir)

# Saves the settings used to test the model 
utils.write_yml(os.path.join(results_dir, 'test_model_settings.yml'), settings)