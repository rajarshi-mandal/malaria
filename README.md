# Modeling Malaria Dynamics for Optimized ITN Allocation

This repository contains data and code used for differential equation modeling of malaria transmission dynamics and optimizing the allocation of Insecticide-Treated Nets (ITNs) using real-world datasets.

## Repository Structure
### Datasets
The following zipped datasets were uploaded and are used in the analysis:

Demographic and Health Surveys.zip

EarthData Giovanni.zip

IR Mapper (Anopheles).zip

Malaria Atlas Project.zip

Multiple Indicator Cluster Surveys.zip

The Global Health Observatory.zip

### Notebooks
Each dataset has a corresponding cleaning notebook:

demographic-and-health-surveys-cleaning.ipynb

earthdata-giovanni-cleaning.ipynb

ir-mapper-anopheles-cleaning.ipynb

malaria-atlas-project-cleaning.ipynb

multiple-indicator-cluster-surveys-cleaning.ipynb

the-global-health-observatory-cleaning.ipynb

### Modeling
mathematical-modelling.ipynb: Implements differential equation-based malaria transmission model.

vae-percent-mortality.ipynb: Experimental notebook on modeling mosquito mortality using Variational Autoencoders (VAE).
