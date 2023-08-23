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
![feature_weights](https://github.com/latypovt/MSTN_GM/assets/119353990/0a690b00-9fd6-4321-838a-afceffabd4bd.png)

***
## 📅 TO DO - this week

1) Writing - introduction, methods, results, discussion
  
  Intro should follow key points listed in the file
  Methods - need a supplemental figure for inclusion of subjects
    Results - need to add HC subjects to the stats
    Discussion - follow the listed structure, need to cite structural MRI papers, fMRI papers, ML papers in TN/chronic pain
    Discussion about MS - open question, need to think

2) Need to add HC subjects to the stats - CamCAN data
3) Get paper done
4) Clean up code

Next version - Mid September, planning virtual meeting.
