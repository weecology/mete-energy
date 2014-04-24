mete-energy
===========

Research project that evaluates the performance of the Maximum Entropy Theory of Ecology (METE) on its four major non-spatial predictions.
The project has been developed by Xiao Xiao, Daniel J. McGlinn, and Ethan P. White.
Our manuscript is available on arXiv (http://arxiv.org/abs/1308.0731). 

Code in this repository replicates all analyses in the manuscript (except bootstrap analysis, which is extremely time-consuming) using a subset of datasets. 
Data included in the repository (under folder ./data/) are redistributed with courtesy of original data owners. 
They are provided to serve the purpose of replicating our results only. 
Readers interested in using the data for any other purposes need to obtain permission from the data owners.

Setup
------------
Requirements: R 2.x or higher, Python 2.x, and the following Python modules: `numpy`, `scipy`, `matplotlib`, `mpl_tookits`, `mpmath`, `multiprocessing`, `pyper`, and `Cython`. 
In addition, the following custom Python modules are also required: `METE` (https://github.com/weecology/METE), `macroecotools` (https://github.com/weecology/macroecotools), and `mete_sads` (https://github.com/weecology/white-etal-2012-ecology).
Note that you will have to navigate to /mete_distributions under module `METE` and run setup.py for Cython to properly compile.

Replicate analyses
------------------
All analyses (except bootstrap) can be replicated by running the following command from the command line: 

`python run_analyses.py`.

By default, figures will be saved to the home directory. 
Intermediate output files will be saved to the subdirectory /out_files. 