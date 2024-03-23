# QMS: Representing mutations for predicting cancer drug response

Quantitative mutation scoring (QMS) is a method for distinguishing influential somatic mutations in cancer drug response models. Somatic mutations present across a tumor genome can influence its response to therapeutic agents. To capture the effects of these mutations, cancer drug models have represented tumors as a collection of gene mutation states, where each gene is assigned a single binary indicator denoting the presence/absence of a somatic mutation (1=mutated, 0=wildtype). No distinction is made between the mutations of a gene.

QMS aims to decipher the varying effects mutations have on drug responses. Each tumor mutation is scored by variant effect prediction algorithms, which predict a continuous value (between 0 and 1) denoting the likely impact a mutation has on normal protein function. Instead of a binary indicator, each gene is assigned one or more scores corresponding to its particular mutations. Now, tumors are represented as a collection of gene perturbation states. 

Included are somatic mutation profiles of 702 genes commonly surveyed in clinical cancer gene panels for 1,244 CCLE/DepMap tumor cell lines. QMS profiles are available for the following methods (click on method name to access original publication):
1. [CHASMplus](https://pubmed.ncbi.nlm.nih.gov/31202631/): Scores the likelihood of a SNV to drive cancer. 
2. [VEST4](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3665549/): Scores the likelihood of a somatic mutation to affect protein function.
3. [CADD](https://academic.oup.com/nar/article/47/D1/D886/5146191): Uses evolutionary conservation to predict deleteriousness of a mutation to protein function. 

Also provided is code for training, testing, interpreting, and visualizing neural networks that model tumor cell line responses to 24 precision oncology drugs and 6 chemotherapeutic agents.

# Environment set up for training and testing QMS models

Neural network training/testing scripts require the following environmental setup:

* Hardware required for training a new model
    - GPU server with CUDA>=11.7 installed
    
* Software
    - Python>=3.10
    - Anaconda
        - Information for installing Anaconda can be found [in the following link]( https://docs.conda.io/projects/conda/en/latest/user-guide/install/)
    - PyTorch
        - PyTorch version used to train our models was 1.13.1
        - Information for installing Pytorch available at (https://pytorch.org/get-started/)
        - Model interpretation uses the torch.tensor.register_hooks method to collect gradients, which is currently not available in PyTorch>=2.0 (PyTorch says register_hooks will work as they finish development of PyTorch 2.0)
    - networkx
    - numpy
    - pandas
    - matplotlib
    - plotly
    - scipy
    - pyyaml

* Set up a virtual environment with Anaconda
    - If you are training a new model or testing the pre-trained models with a GPU server, run the following command to set up a virtual environment (qms):
    """
    conda env create -f conda-env/qms_env.yml
    """
    
# QMS model files
To train and test QMS models, please ensure you have the required files and Python packages. We recommend creating a virtual environment with the qms_env.yml file to ensure all dependencies are met.

1. Cell feature files: Somatic mutation profiles of tumor cell lines, stored as a 2-dimensional array \(1,244 cell lines x 702 genes\). Each row is the somatic mutation profile of a tumor cell line, with 702 comma-delimited values corresponding to the mutation information of each gene. Row indexes match the cell2ind.txt file. Column indexes match the gene2ind.txt file. In cases where a gene has multiple mutations, the largest mutation score was assigned. Unmutated genes are assigned a 0 in all profiles.
    - binary.csv: Each gene is assigned as a binary mutation status indicator \(0=wildtype, 1=mutated\).
    - chasmplus.csv: Each gene is assigned the CHASMplus score of its mutation\(s\).
    - vest.csv: Each gene is assigned the VEST4 score of its mutation\(s\).
    - cadd.csv: Each gene is assigned the CADD score of its mutation\(s\).
    - braf_v600.csv: The BRAF entry of cell lines containing a BRAF V600X mutation are assigned a 1, all other genes in all other cell lines are assigned a 0. This vector is used to benchmark the performance QMS models vs. the knowledge of a precision oncology biomarker.
    - not_scored.csv: Each gene is assigned a binary mutation status indicator if its mutation was not scored by any of the QMS methods \(0=wildtype, 1=mutation not scored by CHASMplus, VEST4, or CADD\).
    
2. Data files: Needed to run train and test functions
    - cell2ind.txt: Tab-delimited file where the first column contains the row index number of cells in the feature files, and the second column is the cell line name used by CCLE/DepMap.
    - compound_info.csv: Comma-delimited file containing information about various drugs.  
    - drug2ind.txt: Tab-delimited file where the first column contains the index value of drugs, and the second column is the SMILES string. Drugs are not explicitly encoded by our single-drug models, but this is a useful feature for tracking the drugs used in various models and train/test datasets.
    - gene2ind.txt: Tab-delimited file where the first column contains the column index number of genes in the feature files, and the second column is the gene name.
    - name2drug.txt: Tab-delimited file where the first column is the name of a drug and the second column is its SMILES string. 
    - ont.txt: Tab-delimited file specifying the architecture layout of the neural network. 
    
3. Code: Python source code files for training and testing QMS models.
    - qms_nn.py: File with the PyTorch neural network class.
    - test_qms_nn.py: Predict drug responses with a pre-trained model. 
    - train_qms_nn.py: Train a new drug response model. 
    - dataloaders.py: Various classes and methods to load and configure data features to train QMS models.
    - utils.py: Various utility functions used in train/test files. 
    
4. Drug response data: Tumor cell line drug response data. Each row contains cell, drug, and response information: Cells are represented to by their CCLE cell line names and correspond to  those in the gene2ind.txt file; Drugs are represented as SMILES stings and correspond to those in the drug2ind.txt file; Response is the continuous value area under the dose response curve (AUDRC). Data for 17 precision oncology drugs has been provided as pre-partitioned nested five-fold cross validation sets. Samples have been approtioned by cell lines into 70%/15%/15% train/validation/test splits. Hyperparameter selection was performed with the train/validation sets of inner nested loop. The final models were trained on the combined train/validation sets of the inner loop and tested on the holdout test set of the outer loop. The file naming convention is:
    - \<\partition\>\_\<\drug\>\_nested_\<\outer_nested_loop\>\_\<\inner_nested_loop\>\.tsv
    - Partition: Train, val, or test set.
    - Drug: Drug name (lowercase).
    - Outer nested loop: Integer value of the outer nested loop.
    - Inner nested loop: Integer value of the inner nested loop. Files where this value is `full` are the combined train/validation sets of the inner loop. This value is 1 for all test files. 
    
# Run a pre-trained model
To run a pre-trained model, execute the following:
'''
python -u code/train_qms_nn.py -model_yml <filepath_to_model_settings.yml>
'''

An example bash script (`commandline_test_qms_nn.sh`) is provided in the `scripts` directory. 

Pretrained models: 
- Available for 5 drugs: Dabrafenib, trametinib, ponatinib, palbociclib, and osimertinib) 
- Located in the `pretrained_models` directory
- Each drug has models for QMS and binary features (10 total)
    - 5 QMS feature models corresponding to the 5 final models of the outer nested loop. 
    - 5 binary feature models (as above)
- Model names include the exact QMS features used to train the final models (binNS is the `not_scored.csv` vector)

Model settings are stored in .yml files and are loaded automatically to run model. Make sure to use the correct settings file for the corresponding model. These settings are:
- batch_size: Number of samples per mini-batch.
- cuda_id: Integer specifying specific GPU if running on a multi-gpu computing framework. Default is 0. 
- dropout: Dropout probability (float).
- features: List of feature file names to run with model. Below specifies a configuration with CHASMplus, VEST4, and CADD QMS features (with additional not_scored vector).
    - chasmplus.csv
    - vest.csv
    - cadd.csv
    - not_scored.csv
- keep_training_n: Terminate training if no improvement to validation set loss after this number of epochs.
- lr: Learning rate (float; already optimized).
- model_path: Path to saved model. Can use absolute path or path relative to the settings file.
- results_dir: Specifies path to directory to save results. Can be absolute path or path relative to the settings file.
- save_grads: Bool to save test set gradients. Saved in a folder called `gradients` located within the results_dir.
- save_hiddens: Bool to save test set hidden activation states. Saved in a folder called `hiddens` located within the results_dir.
- save_model: Bool to save model. Saved as `model.pt` within the results_dir.
- test_file: Filename of the test set located in drug_response_data directory.
- weight_decay: Weight decay lambda (float).
    
Settings files are provided to test pre-trained models. Specifying a unique results_dir for each test is recommended to prevent overwriting previous tests. After running, a new settings file is saved in the results_dir


# Train a new model
To train a new model, execute the following:
'''
python -u code/train_qms_nn.py -model_yml \<\filepath_to_model_settings.yml\>\
'''

An example bash script (`commandline_train_qms_nn.sh`) is provided in the `scripts` directory.

Model settings are stored in .yml files and are loaded automatically to run model. Make sure to use the correct settings file for the corresponding model. These settings are:
- batch_size: Number of samples per mini-batch.
- cuda_id: Integer specifying specific GPU if running on a multi-gpu computing framework. Default is 0. 
- dropout: Dropout probability (float).
- features: List of feature file names to run with model. Below specifies a configuration with CHASMplus, VEST4, and CADD QMS features (with additional not_scored vector).
    - chasmplus.csv
    - vest.csv
    - cadd.csv
    - not_scored.csv
- keep_training_n: Terminate training if no improvement to validation set loss after this number of epochs.
- lr: Learning rate (float; already optimized).
- model_path: Path to saved model. Can use absolute path or path relative to the settings file.
- results_dir: Specifies path to directory to save results. Can be absolute path or path relative to the settings file.
- save_grads: Bool to save test set gradients. Saved in a folder called `gradients` located within the results_dir.
- save_hiddens: Bool to save test set hidden activation states. Saved in a folder called `hiddens` located within the results_dir.
- save_model: Bool to save model. Saved as `model.pt` within the results_dir.
- train_file: Filename of the training set located in drug_response_data directory.
- val_file: Filename of the validation set located in drug_response_data directory.
- weight_decay: Weight decay lambda (float).
    
Settings files are provided with the exact configurations used to train the pre-trained models. When training new models, specifying a unique results_dir for each test is recommended to prevent overwriting previous models. 

    