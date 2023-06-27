# MSTN_GM

Summer project - University of Toronto, IMS
Using Freesurfer parcellation to predict TN pain in MS subjects

### Structure
tSNE - performs dimensionality reduction and visualizes the data structure
model - runs SVM with sequential feature selection to predict TN pain in MS

# Latest results:
Mean train accuracy 99.4%
Mean test accuracy 90.2%


# TO DO - this week


1) tSNE - same story - EDSS (need to add EDSS to the demographics dataframe, run data preprocessing and do tsne vis.)
2) Slides with intro, objectives, methods (description of everything - model, sfs, nested CV, how it works etc.), results, figures, interpretation
3) Important predictors - a short (1-2 pages) review on all structures, how previous research in chronic pain/MS shown these regions being affected. Check lab papers as well
4) Start writing documentation in this file - use Markdown (guide: https://www.markdownguide.org/basic-syntax/)
5) Draft of research proposal
6) Reading on PyTorch (https://machinelearningmastery.com/pytorch-tutorial-develop-deep-learning-models/ and https://blog.paperspace.com/ultimate-guide-to-pytorch/) - pay attention to the dataset and data loading functions, training loop, model construction, evaluation etc. 
7) Reading and tutorials: MONAI (https://github.com/Project-MONAI/tutorials and https://docs.monai.io/en/stable/) - look at the 3d_classification tutorials.
