# Quinten Health Project

[![pylint]()]()

### Explore patients’ feedbacks on their drug treatment

The aim of the project is to explore and better understand patient feedback on our treatments. We have already collected around 300 patient comments on publicly available websites related to our treatment for Crohn's disease and ulcerative colitis.
We expect to be able to extract key themes from this list of patient comments and gain more insight into what our patients think about our solution.
The project also aims to demonstrate, as a proof of concept, that any new comments available about the same treatment can be effectively associated with one of the available topics.

## Dataset details
The comments dataset consists of comments from patients about their drug treatment, each associated with:

- The name of the medication taken (associated to a disease)
- A rate from 1 to 10

## Setting up the project
```
# Create a virtual environment and activate it
conda create --name health_env python=3.8
conda activate health_env

# Install necessary requirements
pip install -r requirements.txt

# For version control please use nbstripout it will strip the outputs of the notebooks and track only the code cells
nbstripout --install

# Copy the csv dataset into the data folder
```
