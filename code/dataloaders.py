import sys, os, random

import numpy as np
import pandas as pd

from multiprocessing import Pool
from joblib import Parallel, delayed

import scipy.stats as stats

import networkx as nx
import networkx.algorithms.components.connected as nxacc
import networkx.algorithms.dag as nxadag



class LoadFeatures():
    """
    Loads and stores input feature vectors from /qms/features/ . Each feature is stored as a numpy array of size n_cells x n_genes, accessed as a dict like structure (eg self['chasmplus'] returns chasmplus array)Provide filenames of features without .csv extension. Object keeps the order of features: chasmplus, vest, cadd, binary, not_scored
    
    Specify desired cell line names or gene names to return a feature array with only those cells/genes using the make_input_vector() method
    
    Methods:
    make_input_vector(cells=cells, genes=genes)
    
    Default available features:
    * chasmplus
    * vest
    * cadd
    * binary
    * not_scored
    * braf_v600
    * skcm2012_chasmplus
    * skcm2012_vest
    * skcm2012_cadd
    * skcm2012_binary
    * skcm2012_not_scored
    
    """
    
    def __init__(self, features, cells=[], genes=[]):
        assert len(features) > 0, 'no feature names provided to LoadFeatures'
        # args is a list of feature names
        self.data_dir = os.path.abspath('../data')
        self.feat_dir = os.path.abspath('../features')
        
        #self.name2vec = name2vec
        self.gene2id = self.load_gene2id()
        self.cell2id = self.load_cell2id()
        self.drug2id = self.load_drug2id()
        self.id2gene = {int(v):k for k,v in self.gene2id.items()}
        self.id2cell = {int(v):k for k,v in self.cell2id.items()}
        self.id2drug = {int(v):k for k,v in self.drug2id.items()}
        
        # Maintains a default feature order
        self.order = ['chasmplus', 'vest', 'cadd', 'binary', 'not_scored', ]
        self.order = self.order + ['skcm2012_'+x for x in self.order] + ['braf_v600']
        
        # Keeps feature order consistent
        ordered_args = [x for x in self.order if x in features]
        
        self.feature_names = [x for x in ordered_args]
        
        for arg in ordered_args:
            arg_vec = self._load_vec_file(arg, cells=cells, genes=genes)
            self.__dict__.update({arg:arg_vec})
    
    def __getitem__(self, key):
        return self.__dict__[key]
        
    def _load_vec_file(self, vec_name, cells=[], genes=[]):
        feat_path = os.path.join(self.feat_dir, vec_name+'.csv')
        feat = np.genfromtxt(feat_path, delimiter=',')
        if cells:
            cell_ids = [self.cell2id[c] for c in cells]
            feat = feat[cell_ids, :]
        if genes:
            gene_ids = [self.gene2id[g] for g in genes]
            feat = feat[:, gene_ids]
            
        print(f'Loaded {vec_name}\tshape {feat.shape}')
            
        return feat
        
    def load_gene2id(self):
        filepath = os.path.join(self.data_dir, 'gene2ind.txt')
        return self._read_mapping_file(filepath)
    
    def load_cell2id(self):
        filepath = os.path.join(self.data_dir, 'cell2ind.txt')
        return self._read_mapping_file(filepath)
    
    def load_drug2id(self):
        filepath = os.path.join(self.data_dir, 'drug2ind.txt')
        return self._read_mapping_file(filepath)
    
    
    def _read_mapping_file(self, filepath):
        mapping = {}
        with open(filepath, 'r') as f:
            for line in f:
                tokens = line.strip().split('\t')
                mapping[tokens[1]] = int(tokens[0])
        return mapping
    
    def make_model_input(self):
        feats = [self[x] for x in self.feature_names]
        return np.concatenate(feats, axis=1)
    
    

            

class LoadOntology():
    """
    Loads NN's architecture. Tracks relationships between model features for downstream analysis
    """
    def __init__(self, ontology_file='ont.txt'):
        self.data_dir = os.path.abspath('../data')
        print('data dir', self.data_dir)
        
        self.ont_filepath = os.path.join(self.data_dir, ontology_file)

        self.genes = self.load_genes()
        
        dG, root, term_size_map, term_direct_gene_map, gene_2_term = self.load()
        
        self.dG = dG
        self.root = root
        self.term_size_map = term_size_map
        self.term2genes = term_direct_gene_map
        self.gene2terms = gene_2_term
        
        term2layer, layer2terms = self.make_layer_maps()
        self.term2layer = term2layer
        self.layer2terms = layer2terms
        
        self.go2name = self.load_go_descriptions()
        
        self.all = (dG, root, term_size_map, term_direct_gene_map, gene_2_term)
    
    def load_genes(self):
        filepath = os.path.join(self.data_dir, 'gene2ind.txt')
        df = pd.read_csv(filepath, sep='\t', names=['ind','gene'])
        return [x for x in df['gene']]
    
    def load(self):
        
        dG = nx.DiGraph()
        gene_2_term = {}
        term_direct_gene_map = {}
        term_size_map = {}

        with open(self.ont_filepath, 'r') as file_handle:
            gene_set = set()
            for line in file_handle:
                line = line.rstrip().split()

                if line[2] == 'default':
                    dG.add_edge(line[0], line[1])
                else:
                    if line[1] not in self.genes:
                        continue

                    if line[0] not in term_direct_gene_map:
                        term_direct_gene_map[line[0]] = set()

                    if line[1] not in gene_2_term:
                        gene_2_term[line[1]] = set()

                    # Collect the "child" of genes at position 0
                    term_direct_gene_map[line[0]].add(line[1])

                    gene_2_term[line[1]].add(line[0])

                    gene_set.add(line[1])

            print('There are', len(gene_set), 'genes')

        for term in dG.nodes():
            term_gene_set = set()
            if term in term_direct_gene_map:
                term_gene_set = term_direct_gene_map[term]

            deslist = nxadag.descendants(dG, term)

            for child in deslist:
                if child in term_direct_gene_map:
                    term_gene_set = term_gene_set | term_direct_gene_map[child]

            if len(term_gene_set) == 0:
                print('There are empty terms, please delete term:', term)
                sys.exit(1)
            else:
                term_size_map[term] = len(term_gene_set)

        leaves = [n for n in dG.nodes if dG.in_degree(n) == 0]

        root = leaves[0]

        uG = dG.to_undirected()
        connected_subG_list = list(nxacc.connected_components(uG))

        print('There are', len(leaves), 'roots:', leaves[0])
        print('There are', len(dG.nodes()), 'terms')
        print('There are', len(connected_subG_list), 'connected componenets')

        if len(leaves) > 1:
            print('There are more than 1 root of ontology. Please use only one root.')
            sys.exit(1)
        if len(connected_subG_list) > 1:
            print('There are more than connected components. Please connect them.')
            sys.exit(1)

        return dG, root, term_size_map, term_direct_gene_map, gene_2_term
    
    def make_layer_maps(self):
        dGl = self.dG.copy()

        term2layer = {}
        layer2terms = {}
        i = 1
        while True:
            leaves = [n for n in dGl.nodes() if dGl.in_degree(n) == 0]

            if len(leaves) == 0:
                break

            # leaves are only terms connected to genes
            layer2terms[i] = leaves

            for term in leaves:
                term2layer[term] = i

            i += 1

            dGl.remove_nodes_from(leaves)
            
        return term2layer, layer2terms
    
    def load_go_descriptions(self):
        go_2_name_file = os.path.join(self.data_dir, 'term_descriptions.txt')
        go_2_name_df = pd.read_csv(go_2_name_file, sep='\t', names=['def', 'name', 'term'], skiprows=1)

        go_2_name_file_2 = '/cellar/users/pgwall/DrugCell/Models/super_model/nest_vnn/validation/go_term_annotations.tsv'
        go_2_name_df_2 = pd.read_csv(go_2_name_file_2, sep='\t', names=['term','name','process','def'])

        go_2_name = dict(zip(go_2_name_df['term'].tolist(), go_2_name_df['name'].tolist()))
        go_2_name_2 = dict(zip(go_2_name_df_2['term'].tolist(), go_2_name_df_2['name'].tolist()))

        go_2_name.update(go_2_name_2)
    
        return go_2_name
    
    def add_genes(self):
        """
        Returns copy of self.dG annotated with self.genes 
        """
        dGg = self.dG.copy()

        for gene, terms in self.gene2terms.items():
            for term in terms:
                dGg.add_edge(term, gene)
                
        return dGg
    
    def add_features(self, dG_genes, features):
        """
        Returns copy of dG_genes annotated with features
        """
        dGgf= dG_genes.copy()
        
        if not isinstance(features, list):
            features = [features]
            
        gene_nodes = [x for x in dGgf.nodes() if x in self.gene2terms]

        for gene in gene_nodes:
            for feat in features:
                feat_str = '_'.join([gene, feat])
                dGgf.add_edge(gene, feat_str)
                
        return dGgf


    

    



