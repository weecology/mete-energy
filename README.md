mete-energy
===========

Research project that evaluates the performance of the Maximum Entropy Theory of Ecology (METE) on its four major non-spatial predictions.
The project has been developed by Xiao Xiao, Daniel J. McGlinn, and Ethan P. White.
Our manuscript is available on arXiv (http://arxiv.org/abs/1308.0731). 

Code in this repository replicates all analyses in the manuscript using a subset of datasets. 

Data included in the repository (under folder ./data/) are redistributed with courtesy of original data owners. 
They are a processed subset of the original datasets specifically designed for the replication of our analyses. 
They will not be updated to reflect additional data collection or corrections to the data made after archiving.
Readers interested in using the data for purposes other than replicating our analyses are advised to obtain the raw data from the original source.

Setup
------------
Requirements: R 2.x or higher, Python 2.x, and the following Python modules: `numpy`, `scipy`, `matplotlib`, `mpl_tookits`, `mpmath`, `multiprocessing`, `pyper`, and `Cython`. 
In addition, the following custom Python modules are also required: `METE` (https://github.com/weecology/METE), `macroecotools` (https://github.com/weecology/macroecotools), and `mete_sads` (https://github.com/weecology/white-etal-2012-ecology).
Note that you will have to navigate to /mete_distributions under module `METE` and run `setup.py` (from the command line: `python setup.py`) for Cython to properly compile.

Replicate analyses
------------------
All analyses (except Monte Carlo and bootstrap) can be replicated by running the following command from the command line: 

`python run_analyses.py`

The Monte Carlo analysis (Appendix D) and the bootstrap (Appendix E), both of which are time-consuming, can be replicated by running the following command from the command line:

`python run_analyses_bootstrap.py`

By default, figures will be saved to the home directory. 
Intermediate output files will be saved to the subdirectory /out_files/. 
