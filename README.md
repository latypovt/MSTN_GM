# 🧠 MSTN_GM 

*Summer project - University of Toronto, [IMS SURP](https://ims.utoronto.ca/surp)*  
Abigail Wolfensohn, Timur Latypov, [Hodaie Lab](https://www.hodaielab.com/)
  
***Primary objective:** Using Freesurfer parcellation to predict TN pain in MS subjects*  


***
## 💾 Structure
* **data_preprocessing.py -** retrieves and reorganizes raw MS and MS-TN demographic and featural data so that it can be used by the machine learning model  
* **tSNE.py -** performs dimensionality reduction and visualizes the data structure  
* **model.py -** runs SVM with sequential feature selection to predict TN pain in MS; nested k-fold cross-validation
* **graphs.ipynb -** runs graphic representations of data gathered by the model, as well as an independent t-test
* **stats -** folder for csv files containing featural (from Freesurfer segmentation) and demographic data for each subject
* **utils -** folder for other important files to be used by the model and graphing notebook
* **out -** folder for output files

***
## 📌 Latest results:
    Mean train accuracy 99.6%  
    Mean test accuracy 90.2%

***
## 📅 TO DO - this week

1) tSNE - same story - EDSS (need to add EDSS to the demographics dataframe, run data preprocessing and do tsne vis.) - EDSS - step, duration of MS, pain/no pain, age, sex ✅
2) Slides with intro, objectives, methods (description of everything - model, sfs, nested CV, how it works etc.), results, figures, interpretation ✅
3) Important predictors - a short (1-2 pages) review on all structures, how previous research in chronic pain/MS shown these regions being affected. Check lab papers as well
4) Start writing documentation in this file - use Markdown (guide: https://www.markdownguide.org/basic-syntax/) ✅
5) Draft of research proposal
6) Reading on PyTorch (https://machinelearningmastery.com/pytorch-tutorial-develop-deep-learning-models/ and https://blog.paperspace.com/ultimate-guide-to-pytorch/) - pay attention to the dataset and data loading functions, training loop, model construction, evaluation etc. 
7) Reading and tutorials: MONAI (https://github.com/Project-MONAI/tutorials and https://docs.monai.io/en/stable/) - look at the 3d_classification tutorials.
