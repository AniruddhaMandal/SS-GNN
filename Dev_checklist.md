- [] Reproducibility
    - [x] Datasplit
    - [x] Model Parameter 
    - [] ss-gnn
- [x] LRGB Baseline
- [x] Add GNN-type in VANILLA
- [] Add pytest 
    - [x] Reproducibility
    - [] Data sanity
    - [] Model sanity
- [x] SS-GNN 
- [] Load from checkpoint
- [x] Add instruction to README file
    - [] Add config file doc
- [x] add default config file where every 
     option for all arguments are present
- [] Add support for multiple metric
- [] Add QM9
- [] WandB
    - [x] cli to turn off. 
    - [] resolve entity name 
    - [] wandb projcet arg for main.py 
    - [] parameter names in wandb


## workflow
1. choose a dataset. 
2. choose a sampler
3. Fix k-size.
4. Test on many graph of the dataset.
    - check coverage
    - check cv
    - check sampler speed for processing the entire dataset.
    - coverage on smaller m-per-graph.
5. Check sampler failures on the dataset.
6. *Finaly* performance of ss-gnn on the dataset 
        with the choosen sampler
    