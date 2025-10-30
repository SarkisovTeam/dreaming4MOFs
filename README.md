# dreaming4MOFs

<p align="center">
  <img src="./figures/dreaming.PNG" style="width: 25%;" />
</p>

[![DOI](https://zenodo.org/badge/818152690.svg)](https://doi.org/10.5281/zenodo.15089008)


This repo is intended as a documentation of the simulation experiments conducted in the main article of: [Inverse design of metal-organic frameworks using deep dreaming approaches](https://doi.org/10.1038/s41467-025-59952-3). 


Note:
-----
The scripts are provided as-is, and are not guaranteed to work without the required dependencies or with a different structure from that used in our work. If you have any questions, please contact the corresponding author of the article.

## Repo contents
- [deep_dream_src](https://github.com/SarkisovTeam/dreaming4MOFs/tree/main/deep_dream_src): source code and functions used to: (1) [define the machine learning architecture](https://github.com/SarkisovTeam/dreaming4MOFs/blob/main/deep_dream_src/nn_functions.py), (2) [featurise and prepare the training dataset for the forward process](https://github.com/SarkisovTeam/dreaming4MOFs/blob/main/deep_dream_src/nn_functions.py), (3) [perform dreaming experiments](https://github.com/SarkisovTeam/dreaming4MOFs/blob/main/deep_dream_src/dreaming_functions.py), which relies on [encoding (tokenizing) and decoding (converting back) MOF strings](https://github.com/SarkisovTeam/dreaming4MOFs/blob/main/deep_dream_src/utils.py), and finally (4) [reconstruct MOFs](https://github.com/SarkisovTeam/dreaming4MOFs/blob/main/deep_dream_src/sbu_functions.py) 
- [encode_MOFs](https://github.com/SarkisovTeam/dreaming4MOFs/tree/main/encode_MOFs): contains jupyter notebook to encode MOFs using pormake building blocks and topology codes.
- [deep_dreaming_experiments](https://github.com/SarkisovTeam/dreaming4MOFs/tree/main/deep_dreaming_experiments): contains the jupyter notebooks and model results used to create the figures in the main article. 
- [train_models](https://github.com/SarkisovTeam/dreaming4MOFs/tree/main/train_models): contains the jupyter notebooks to train the models and visualise the parity plots, along with the pretrained models. It also contains the cif files and raw data used to train the models.
- [construct_MOFs](https://github.com/SarkisovTeam/dreaming4MOFs/tree/main/construct_MOFs): codes used to construct MOFs from their MOF string representations. 

## System Requirements

### Hardware Requirements

To train the machine learning model, we used a single NVIDIA v100 SXM2 16GB GPU (Volta architecture – hardware v7.0, compute architecture `sm_70`). Model training took approximately 2 hours on average. However, there is opportunity to speed up this process using sequence masking, as we only utilised batch sizes of 1 to deal with variable length sequences while training the model. 

To perform deep dreaming experiments, a standard computer / laptop with sufficient RAM to support the in-memory operations is all that is required. For example, to run the jupyter notebooks contained in `/deep_dreaming_experiments/`, a laptop with 32 GB of RAM, 14 CPU cores, and a 12th Gen Intel(R) Core(TM) i7-12700H 2.30 GHz processor was used. A single deep dreaming experiment (optimising the linker representation over 5,000 epochs) using this hardware setup takes approximately 2 minutes to complete. Dreaming experiments were only conducted on hardware with these specifications, and so we recommend performing dreaming experiments using CPU rather than GPU. 

### Software Requirements

The package has been tested on Windows 10 (64-bit, v.22H2).

## Dependencies

An `environment.yml` file is available for setting up a new Python environment. Most of the required dependencies are handled in this file, and can be executed using the following command (which should only take a few minutes):

```
conda env create -f environment.yml
```

However, the `group selfies` library must be installed separately. Instructions on how to install this package are available from the `group selfies` [repository](https://github.com/aspuru-guzik-group/group-selfies.git) (we used v1.0.0 to develop the code in this repo).

The code is primarily built upon the following libraries: `pytorch`, `numpy`, `pandas`, [selfies](https://github.com/aspuru-guzik-group/selfies.git), [group selfies](https://github.com/aspuru-guzik-group/group-selfies.git), `rdkit`, `scikit-learn`, `ase`, [PORMAKE](https://github.com/Sangwon91/PORMAKE.git)

## 1. Training deep dreaming models

1.	To train a deep dreaming model, go to the directory `/train_models/` and run the `train_model.ipynb` jupyter notebook. It is recommended to train these models on a GPU. 
2.	Pre-trained deep dreaming models are available in the `/train_models/mof_saved_models/` directory.
3.	To visualise the parity plots for these models, run the `/train_models/plot_performance.ipynb` jupyter notebook. 


## 2. Deep dreaming experiments

1. Go to the `/deep_dreaming_experiments/` directory and explore the jupyter notebooks for the property case studies explored in the main article above (pore volume, $c_p$, and $Q_{CO_2}$ - $S_{CO_2/N_2}$). The studies make use of the pretrained models avaialable in `/train_models/mof_saved_models/`
2. Some of these studies rely on optimising seed distributions, which are contained in the `/deep_dreaming_experiments/seeds/` directory.
3. Deep dreaming results are stored in the `/dream_results/` subdirectory for each case study, and are visualised within their respective jupyter notebooks. 

## 3. Construct MOFs
1. Go to the `/construct_MOFs/` directory and run the `construct_dreamed_mofs.ipynb` jupyter notebook. 
2. To construct MOFs, you need the linker SMILES string (or the file path to a pre-generated building block XYZ file), the PORMAKE topology string, and the PORMAKE node string.

## 4. Property determination
After constructing MOFs in step 3 above, we can utilise some open-source softwares to determine their properties (both for training and dreaming).

1. For structural properties we used [Zeo++v.0.3](https://zeoplusplus.org/) with a probe radius of 1.86 Å and high accuracy settings. 
2. The heat capacity is calculated using the machine learning model of Moosavi et al. [1] The coding implementation can be found on their [github repo.](https://github.com/SeyedMohamadMoosavi/tools-cp-porousmat)
3. For heats of $CO_2$ adsorption and $CO_2$ / $N_2$ Henry's selectivity (i.e., $Q_{CO_2}$ - $S_{CO_2/N_2}$), we use the [RASPA2](https://github.com/iRASPA/RASPA2) simulation package (as implemented in the [mofdscribe](https://github.com/kjappelbaum/mofdscribe) package [2] [`v0.0.9.dev0`]). 

Additional details on these properties can be found in the methods section [here](https://chemrxiv.org/engage/chemrxiv/article-details/6628ea2721291e5d1d93a83e).

## Citations
[1] Moosavi, S.M., Novotny, B.Á., Ongari, D. et al. A data-science approach to predict the heat capacity of nanoporous materials. Nat. Mater. 21, 1419–1425 (2022).

[2] Jablonka, Kevin Maik, Andrew S. Rosen, Aditi S. Krishnapriyan, and Berend Smit. An ecosystem for digital reticular chemistry. ACS Cent. Sci. 563-581 (2023).