import os, yaml

import numpy as np
import pandas as pd
import scipy.stats as stats

import torch



def write_yml(filepath, settings):
    with open(filepath, 'w') as f:
        yaml.dump(settings, f, default_flow_style=False)
        
        
def read_yml(filepath):
    with open(filepath, 'r') as f:
        settings = yaml.safe_load(f)
    return settings


def build_input_features(model_input, input_data, label):
    n_samp_batch = input_data.shape[0]
    n_feats = model_input.shape[1]

    feature = np.zeros((n_samp_batch, n_feats))

    for i in range(n_samp_batch):
        feature[i] = model_input[int(input_data[i,0])]

    return feature, label


def create_results_directory(settings):
    if not settings['results_dir']:
        try:
            results_dir = settings['train_file'].split('.')[0]+'_results'
        except:
            results_dir = settings['test_file'].split('.')[0]+'_results'
    else:
        results_dir = settings['results_dir']
        
        if '..' not in results_dir:
            results_dir = os.path.abspath('../'+results_dir)
            
    os.makedirs(results_dir, exist_ok=True)
    settings['results_dir'] = results_dir
    return results_dir


def load_drug_data(file, features):
    """
    Loads drug data file, returns 2 numpy arrays: [cells, drugs] 
    """
    train_test_dir = os.path.abspath('../drug_response_data')
    filepath = os.path.join(train_test_dir, file)
    data = pd.read_csv(filepath, names=['cell','drug','resp'], sep='\t')
    data = data.replace(to_replace={'cell':features.cell2id})
    data = data.replace(to_replace={'drug':features.drug2id})
    data = data.to_numpy()
    return data[:, :2], data[:,2].reshape(-1,1)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) 

               
def pearson_corr(x, y):
    xx = x - torch.mean(x)
    yy = y - torch.mean(y)
    return torch.sum(xx * yy) / (torch.norm(xx, 2) * torch.norm(yy, 2))


def save_grad(term, saved_grads):
    def savegrad_hook(grad):
            saved_grads[term] = grad
    return savegrad_hook


def save_gradients(aux_out_map, hidden_maps, model_dir):
    gradients_dir = os.path.join(model_dir, 'gradients')
    
    if not os.path.exists(gradients_dir):
        os.makedirs(gradients_dir)

    # Register hook to save gradients
    saved_grads = {}
    for hidden_map in hidden_maps:
        for term in hidden_map:
            hidden_map[term].register_hook(save_grad(term, saved_grads)) 
    
    # fire gradients
    aux_out_map['final'].backward(torch.ones_like(aux_out_map['final']))
    
    # Save gene gradients
    for term, grad in saved_grads.items():
        filepath = os.path.join(gradients_dir, term+'.grad')
        with open(filepath, 'ab') as f:
            np.savetxt(f, grad.data.cpu().numpy(), '%.4e')
    
    # Saves the input features for each gene for downstream analyses
    for term, inpt in hidden_maps[-1].items():
        filepath = os.path.join(gradients_dir, term+'.input')
        with open(filepath, 'ab') as f: #
            np.savetxt(f, inpt.detach().cpu().numpy(), '%.4e')
            
            
            
def save_predictions(mode, data, preds, model_dir):
        
    filepath = os.path.join(model_dir, mode+'.predict')
    
    # cell, drug, resp, pred
    data_stack = np.hstack((data[0], data[1], preds))

    data = pd.DataFrame(data_stack, columns=['cell', 'drug', 'resp', 'pred'])
    data.loc[:, 'cell'] = data['cell'].astype(int)
    data.loc[:, 'drug'] = data['drug'].astype(int)

    if os.path.exists(filepath):
        data.to_csv(filepath, sep='\t', index=None, header=None, mode='a')
    else:
        data.to_csv(filepath, sep='\t', index=None)
        
        

def save_results(settings_dict, best_results, results_dir):
    settings_dict.update(best_results)
    
    filepath = os.path.join(results_dir, 'results.tsv')
    
    df = pd.DataFrame.from_dict({k:[v] for k,v in settings_dict.items()})
    
    df.to_csv(filepath, index=None, sep='\t')
            
            
# TODO: Needs model dir specified
def save_hiddens(out_map_list, model_dir, writing_mode='w'):
    hiddens_dir = os.path.join(model_dir, 'hiddens')
    
    if not os.path.exists(hiddens_dir):
        os.makedirs(hiddens_dir)

    for hiddens_maps in out_map_list:
        for term, hidden_map in hiddens_maps.items():
            hidden_file = os.path.join(hiddens_dir, term+'.hidden')
            if os.path.exists(hidden_file):
                writing_mode = 'ab'
            with open(hidden_file, writing_mode) as f: #'ab'
                np.savetxt(f, hidden_map.data.cpu().numpy(), '%.5e')
                    
                    
def load_results(model_dir):
    filepath = os.path.join(model_dir, 'results.tsv')
    return pd.read_csv(filepath, sep='\t')



def load_saved_model(qms_nn, model_path, num_hiddens_gene, id2gene, gene_dim, feat_names, ont, dropout=None):
    if '..' in model_path:
        model_path = os.path.abspath(model_path)
    
    model = qms_nn(num_hiddens_gene, dropout, id2gene, gene_dim, feat_names, ont,)
    
    model_state = torch.load(model_path, map_location=torch.device('cpu'))
    
    model.load_state_dict(model_state)
    model.eval()
    
    return model


def set_mode(train_file='', val_file='', test_file=''):
    mode='val'
    if test_file or 'test' in val_file:
        mode='test'
    return mode


def calc_pearsonr_and_ci(preds):
    """
    Calculate Pearson rho for a prediction df with "pred" and "resp" columns. Returns tuple of (rho, ci_lo, ci_hi)
    """
    n = preds.shape[0]

    rho = stats.pearsonr(preds['pred'], preds['resp'])[0]
    
    r_to_z = 0.5*np.log((1+rho)/(1-rho))
    
    ci_lo = r_to_z - 1.96*(1/np.sqrt(n-3))
    ci_hi = r_to_z + 1.96*(1/np.sqrt(n-3))
    
    z_to_r_lo = (np.exp(2*ci_lo) - 1)/(np.exp(2*ci_lo) + 1)
    z_to_r_hi = (np.exp(2*ci_hi) - 1)/(np.exp(2*ci_hi) + 1)
    
    return (rho, z_to_r_lo, z_to_r_hi)


def compare_r_dist(preds1, preds2):
    """
    Calculate Pearson rho for a prediction df with "pred" and "resp" columns. Returns tuple of (rho, ci_lo, ci_hi)
    """
    n1 = preds1.shape[0]
    n2 = preds2.shape[0]

    rho1 = stats.pearsonr(preds1['pred'], preds1['resp'])[0]
    rho2 = stats.pearsonr(preds2['pred'], preds2['resp'])[0]
    
    r_to_z1 = 0.5*np.log((1+rho1)/(1-rho1))
    r_to_z2 = 0.5*np.log((1+rho2)/(1-rho2))
    
    std1 = 1/np.sqrt(n1-3)*np.sqrt(n1)
    std2 = 1/np.sqrt(n2-3)*np.sqrt(n2)
    
    s,p = stats.ttest_ind_from_stats(r_to_z1, std1, n1, r_to_z2, std2, n2, equal_var=True) # alternative='greater'
    
    return s, p


