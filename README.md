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
    Mean train accuracy 99.5%  
    Mean test accuracy 93.4%
![feature_weights](https://github.com/latypovt/MSTN_GM/assets/119353990/0a690b00-9fd6-4321-838a-afceffabd4bd.png)

***
## 📅 Timeline

Manuscript is in preparation.
