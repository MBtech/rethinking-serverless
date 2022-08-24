# Optimizers and Analysis
This directory contains the scripts and notebooks used for optimization and analysis. 

`optimizer` contains the optimization algorithms. For the analysis in the research paper we have only used bayesian optimization in `boskopt.py`

Running the optimizers or doing analysis on the data can take a while (5-30mins+), therefore the jupyter notebooks store the data from the analysis in CSV files so that the data can be loaded and plotted by parts of the notebooks without having to run the whole notebook again. 

## Pre-requisites:
1. Have a setup to run python notebooks. I have used VScode with Jupyter extension that also prompts for missing dependencies to run notebooks
2. Python version install needs to be >=3.8

## Setup 
1. To use the code you need to install the required python modules:
```
cd analysis
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Create an `analysis_data` directory under `analysis` folder. This is where the jupyter notebooks store the analysis data. 
`cd analysis && mkdir -p analysis_data`


## Notebooks:
The following notebooks are available. See each notebook for information about the code available in the notebook:
1. `decoupled-vs-default.ipynb`: contains the comparison of the best configurations found with different resource allocation strategies i.e. proportional CPU strategy, Fixed CPU strategy, Decoupled and Decoupled(m5)
2. `alternative-analysis.ipynb`: contains the code to get the alternative configurations (with different instance types) that have performance with a threshold of the best configuration. 
3. `optimization-across-data.ipynb`: contains evaluation of optimization across different input data. 
4. `hierarchical-mo.ipynb`: contains the script to performance hierarchical multi-objective optimization and plot the results.
5. `multi-objective.ipynb`: contains the script to performance weighted multi-objective optimization and plot the results.