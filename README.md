# Data Augmentation for financial time series



This repository contains modifications to the implementation of data augmentation for financial time series classification.

Paper Link: https://arxiv.org/abs/2010.15111

The dataset corresponds to the returns of the S&P500 constituent stocks from 1990 to present.



### Usage summary

Create environment using the requirements.txt file. The code was developed in Python 3.6 and requires PyTorch\==1.7.1 and tsaug\==0.2.1.

```
$ conda create --name <env> --file requirements.txt
```

To process the data, run the following notebook. It takes S&P500 data from YFinance and builds the train and test splits of data and writes them on a new directory.

```
notebooks\DataPreProcessing.ipynb
```

To model the data, please run the following notebook.

```
notebooks\Modeling.ipynb
```

This will create a folder <dest_dir> with the output of the run which consists of a folder _output_ and inside, one folder per split of data containing the saved model, the evaluation metrics and the predicted best top/bottom 10 stocks to build the portfolios.

##### Optional parameters in modeling file

```

  batch_size BATCH_SIZE
                        batch size (default: 256)
  hidden_dim HIDDEN_DIM
                        hidden dimension of LSTM (default: 25)
  n_layers N_LAYERS   number of layers in the LSTM (default: 1)
  n_epochs N_EPOCHS   number of epochs for training (default: 600)
  init_sp INIT_SP     initial data split (default: 0)
  end_sp END_SP       final data split (default: 1)
  patience PATIENCE   patience for early stopping (default: 10)
  da_method {convolve,pool,jitter,quantize,reverse,timewarp,spawner,scaling,magnitude_warp,window_warp,None}
                        augmentation methods (default: None)
```

### Citation

E. Fons and P. Dawson and X. Zeng and J. Keane and A. Iosifidis, "Evaluating data augmentation for financial time series classification", arXiv, 2020.

```bibtex
@article{fons2020dafin,
title={Evaluating data augmentation for financial time series classification}, 
author={Elizabeth Fons and Paula Dawson and Xiao-jun Zeng and John Keane and Alexandros Iosifidis},
year={2020},
eprint={2010.15111},
archivePrefix={arXiv},
 primaryClass={q-fin.ST}
}
```

