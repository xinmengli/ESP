# Ensembled Spectral Prediction for Metabolite Annotation (ESP)

### Xinmeng Li, Hao Zhu, Li-Ping Liu, Soha Hassoun
#### Department of Computer Science, Tufts University

A key challenge in metabolomics is annotating spectra measured from a biological sample with chemical identities. We improve on prior neural network-based annotation approaches, namely MLP-based [1] and GNN-based [2] approaches. We propose a novel ensemble model to take advantage of both MLP and GNN models. First, the MLP and GNN are enhanced by: 1) multi-tasking on additional data (spectral topic labels obtained using LDA (Latent Dirichlet Allocation) [3], and 2) attention mechanism to capture dependencies among spectra peaks. Next, we create an Ensembled Spectral Prediction (ESP) model that is trained on ranking tasks to generate the average weighted MLP and GNN spectral predictions. Our results, measured in average rank and Rank@K for the test spectra, show remarkable performance gain over existing neural network approaches.


### Given files:
    
- source code files
    - `gnn_sp.py`: gnn model for spectra prediction
    -  `mlp_mt.py`: mlp models for spectra prediction
   -  `utils.py`: helper functions
- `data` folder:
        - `test_inchi_ik_dict.pkl`: inchikey to inchi mapping for candidates
        - `MHfilter_trvate_idx.pkl`: index of M+H precursors
        - `filter_te_idx.pkl`: index of test after filtering
        - `lda_100.npz`: lda topics 
 - `results` folder:
    - `best_model_gnn_e.pt`: best gnn model trained with lda, attention for ensemble model
   - `best_model_mlp_e.pt`: best mlp model trained with lda, attention for ensemble model

    
### Files to run:
1. `data_trvate.py` -> `torch_trvate_1000bin.pkl`, `trvate_idx.pkl`
    - generate full dataset for training, validation, and test data:
        - `torch_trvate_"+str(int(1000/hp['resolution']))+"bin.pkl`
    - to run: 
        - `python data_trvate.py`
2.  `data_tecand.py` -> `torch_tecand_1000bin_te_cand100.pkl`
    - generate file for test candidate torchgeometric: 
        - `torch_tecand_'+str(int(1000/hp['resolution']))+"bin_te_cand"+str(hp['cand_size'])+'_torchgeometric.pkl`
    - to run: 
      - `python data_tecand.py`

3.  `train.py` -> `best_model_gnn_XXX.pt, best_model_mlp_XXX.pt`
    - Train and restore saved GNN and MLP models
    - To run to saved GNN model by loading file `best_model_gnn_e.pt`:
        - `python train.py --cuda 5 --model gnn --model_file_suffix gnn_e --disable_two_step_pred --disable_fingerprint --disable_mt_fingerprint --disable_mt_ontology --correlation_mat_rank 100`
    - To run saved MLP model by loading file `best_model_mlp_e.pt`
        - `python train.py --cuda 5 --model mlp --model_file_suffix mlp_e --disable_two_step_pred --disable_fingerprint --disable_mt_fingerprint --disable_mt_ontology --correlation_mat_rank 100`
    - To run to train new GNN or MLP model, 
      *** REMEMBER TO UN-COMMENT `train(epoch=args.epochs)`***
        - generate file with parameters: best_model_' + args.model_file_suffix + '.pt
        - `python train.py --cuda 5 --model gnn --disable_two_step_pred --disable_fingerprint --disable_mt_fingerprint --disable_mt_ontology --correlation_mat_rank 100 --model_file_suffix gnn_XXX`
### Conda environments:
1. Full list of packages to run `data_trvate.py` and `data_tecand.py` is in env_d.yml,  to run `train.py` is in env.yml

2. Key packages needed 
    - For `data_trvate.py` and `data_tecand.py`:
        - python=3.8
        - dgl=0.6.1
        - pytorch=1.7.1
        - rdkit
    - For `train.py`:
        - python=3.8
        - pytorch=1.9.0
        - pytorch-geometric=1.7.2
        - numpy
        - scikit-learn
        - scipy

#### References
[1] Wei, J.N., Belanger, D., Adams, R.P. and Sculley, D., 2019. Rapid prediction of electron–ionization mass spectrometry using neural networks. ACS central science, 5(4), pp.700-708.

[2] Zhu, H., Liu, L. and Hassoun, S., 2020. Using Graph Neural Networks for Mass Spectrometry Prediction. arXiv preprint arXiv:2010.04661.

[3] van Der Hooft, J.J.J., Wandy, J., Barrett, M.P., Burgess, K.E. and Rogers, S., 2016. Topic modeling for untargeted substructure exploration in metabolomics. Proceedings of the National Academy of Sciences, 113(48), pp.13738-13743.


#### Contact
Xinmeng.li@tufts.edu