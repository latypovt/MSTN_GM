# 🧠 Signatures of chronic pain in multiple sclerosis: a machine learning approach to investigate trigeminal neuralgia 
Scripts and code used for the analysis
  
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
Please check our [manuscript](https://journals.lww.com/pain/fulltext/9900/signatures_of_chronic_pain_in_multiple_sclerosis_.789.aspx) published in PAIN 

Latypov TH, Wolfensohn A, Yakubov R, et al. Signatures of chronic pain in multiple sclerosis: a machine learning approach to investigate trigeminal neuralgia. Pain. Published online December 13, 2024. doi:10.1097/j.pain.0000000000003497

