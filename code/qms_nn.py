import numpy as np

import torch
import torch.utils.data as du
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import argparse, time, gc, sys, os

import networkx as nx
import networkx.algorithms.components.connected as nxacc
import networkx.algorithms.dag as nxadag


class qms_nn(nn.Module):
    # TO DO: 
    def __init__(self,  num_hiddens_gene, dropout, id2gene, gene_dim, features, ont,  **kwargs):
        
        super().__init__()
        
        # Model and vector info ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.features = features
        self.feat_dim = len(features)    # N input features
        self.gene_dim = gene_dim        # N genes

        self.dropout = dropout
        
        # dictionary for GO terms directly connected to gene
        dG, root, _, term_direct_gene_map, _ = ont.all
        self.root = root
        self.term_direct_gene_map = term_direct_gene_map
        
        # dictionary with index:'gene_name' pairs
        self.id2gene = id2gene
        
        # Model architecture info and mosule creation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.num_hiddens_gene  = [num_hiddens_gene]  
        self.num_hiddens_graph = 4
        self.num_hiddens_final = 4 
        
        self.construct_gene_embedding()
        
        self.construct_NN_graph(dG)

        # Add modules for final layer ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.add_module('final_linear_layer', nn.Linear(self.num_hiddens_graph, self.num_hiddens_final))
        self.add_module('final_batchnorm_layer', nn.BatchNorm1d(self.num_hiddens_final))
        self.add_module('final_aux_linear_layer', nn.Linear(self.num_hiddens_final, 1))
        self.add_module('final_linear_layer_output', nn.Linear(1, 1))
        
    
    # Define module construction functions ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def construct_gene_embedding(self):
        """ 
        Iterate through gene index values, Add linear layer for each gene instance that takes in features
        """
        gene_node_count = 0

        for gene in range(self.gene_dim):
            gene = self.id2gene[gene]
            
            input_size = self.feat_dim
            for i, output_size in enumerate(self.num_hiddens_gene):
                
                if self.dropout:
                    self.add_module(str(gene)+'_dropout_layer_'+str(i+1), nn.Dropout(p=self.dropout))
                self.add_module(str(gene)+'_gene_layer_'+str(i+1), nn.Linear(input_size, output_size))
                self.add_module(str(gene)+'_batchnorm_layer_'+str(i+1), nn.BatchNorm1d(output_size))
                
                self.add_module(str(gene)+'_aux_linear_layerA_'+str(i+1), nn.Linear(output_size, 1))
                self.add_module(str(gene)+'_aux_linear_layerB_'+str(i+1), nn.Linear(1, 1))
                    
                input_size = output_size
                
            gene_node_count += 1
            
        print(f'Number of gene nodes created: {gene_node_count}')
        

    # Construct intermediate layers between genes and output 'root'
    def construct_NN_graph(self, dG):

        self.term_layer_list = []  
        self.term_neighbor_map = {}
        ont_node_count = 0

        for term in dG.nodes():
            self.term_neighbor_map[term] = []
            for child in dG.neighbors(term):
                self.term_neighbor_map[term].append(child)

        while True:
            leaves = [n for n in dG.nodes() if dG.out_degree(n) == 0]

            if len(leaves) == 0:
                break

            self.term_layer_list.append(leaves)
     
            for term in leaves:
                input_size = 0

                for child in self.term_neighbor_map[term]:
                    input_size += self.num_hiddens_graph

                # Multiply number_genes * hidden_dim for total feature length 
                if term in self.term_direct_gene_map:
                    input_size += len(self.term_direct_gene_map[term])*self.num_hiddens_gene[-1]

                term_hidden = self.num_hiddens_graph
                if self.dropout:
                    self.add_module(term + '_dropout_layer',  nn.Dropout(p=self.dropout))
                self.add_module(term + '_linear_layer', nn.Linear(input_size, term_hidden))
                self.add_module(term + '_batchnorm_layer', nn.BatchNorm1d(term_hidden))
                self.add_module(term + '_aux_linear_layer1', nn.Linear(term_hidden, 1))
                self.add_module(term + '_aux_linear_layer2', nn.Linear(1, 1))
                
                ont_node_count+=1

            dG.remove_nodes_from(leaves)

    
    # Define helper functions for main forward() class ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def prep_gene_data(self, x, gene_id):
        """
        Makes a new tensor of the input features of a gene. Used to collect input feature gradients
        """
        gene_idxs = [gene_id+(i*self.gene_dim) for i in range(self.feat_dim)]
        gene_inputs = x[:, gene_idxs].clone().detach().requires_grad_(True)
        return gene_inputs      
    
    
    def gene_embedding_forward(self, inputs, gene_name, i):
            
        if self.dropout:
            inputs = self._modules[str(gene_name)+'_dropout_layer_'+str(i+1)](inputs)
        outputs = self._modules[str(gene_name)+'_gene_layer_'+str(i+1)](inputs)
        batchnorm_out = self._modules[str(gene_name)+'_batchnorm_layer_'+str(i+1)](outputs)
        tanh_out = torch.tanh(batchnorm_out)
                
        return tanh_out
    
    
    def gene_aux_layers_forward(self, inputs, gene_name, i):
        aux_layer1_out = torch.tanh(self._modules[str(gene_name)+'_aux_linear_layerA_'+str(i+1)](inputs))
        aux_layer2_out = self._modules[str(gene_name)+'_aux_linear_layerB_'+str(i+1)](aux_layer1_out)
        
        return aux_layer2_out

    
    # Forward ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def forward(self, x):
        
        # Store all activation outputs for GO term IDs and gene index numbers. Ft embeddings stored as '{gene_name}_{feature}'
        input_map = {}
        out_map = {}
        out_map_gene = {}
        out_aux_map = {}

        for idx, gene in self.id2gene.items():
            inputs = self.prep_gene_data(x, idx)

            for i, _ in enumerate(self.num_hiddens_gene):
                gene_str = gene+'_'+str(i+1)

                gene_out = self.gene_embedding_forward(inputs, gene, i)
                gene_aux_out = self.gene_aux_layers_forward(gene_out, gene, i)

                input_map[gene+'_input'] = inputs

                out_map_gene[gene_str] = gene_out
                out_aux_map[gene_str] = gene_aux_out

                inputs = gene_out
                    
        
        for i, layer in enumerate(self.term_layer_list):

            for term in layer:

                child_input_list = []
                
                # Appends any existing direct child outputs from previous forward pass
                for child in self.term_neighbor_map[term]:
                    child_input_list.append(out_map[child])
                
                # If GO_term has direct gene neigbors
                g_idx = str(len(self.num_hiddens_gene))
                if term in self.term_direct_gene_map:
                    child_input_list.extend([out_map_gene[gene+'_'+g_idx] for gene in self.term_direct_gene_map[term]])   
                
                child_input = torch.cat(child_input_list, dim=1)

                term_NN_out = self._modules[term + '_linear_layer'](child_input)
                Tanh_out = torch.tanh(term_NN_out)
                out_map[term] = self._modules[term + '_batchnorm_layer'](Tanh_out)
                
                aux_layer1_out = torch.tanh(self._modules[term + '_aux_linear_layer1'](out_map[term]))
                out_aux_map[term] = self._modules[term + '_aux_linear_layer2'](aux_layer1_out)
                
                term_str_input = term+'_input'
                input_map[term_str_input] = term_NN_out #child_input
        
            
        final_input = out_map[self.root]

        out = self._modules['final_batchnorm_layer'](torch.tanh(self._modules['final_linear_layer'](final_input)))
        out_map['final'] = out

        aux_layer_out = torch.tanh(self._modules['final_aux_linear_layer'](out))
        out_aux_map['final'] = self._modules['final_linear_layer_output'](aux_layer_out)

        
        # Ouputs: 
        # * out_aux_map  --> Predicted AUDRC of samples
        # * out_map_gene --> hidden activation states of gene ne
        # * input map    --> input features of each gene
        return out_aux_map, out_map, out_map_gene, input_map