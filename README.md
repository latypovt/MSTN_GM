# MSTN_GM

Summer project - University of Toronto, IMS
Using Freesurfer parcellation to predict TN pain in MS subjects

### Structure
tSNE - performs dimensionality reduction and visualizes the data structure
model - runs SVM with sequential feature selection to predict TN pain in MS


## TO DO on Monday and upcoming week

1) Update data_preprocessing script: hippocampus, thalamus and amygdala vols  - merging it with cortex data. NB! all subcortical colnames need to be modified  hemisphere prefix (lh_/rh_). 
2) tSNE - same story
3) Univariate statistics between MS ans MS-TN on the important predictors. Load predictor names from the csv - similar to the plots. Run independent t-test on these features, collect p_vals to the list. Correct the list of pvals for the multiple tests ("FDR"). Save output as csv with corrected p, plot the boxplot, with hue being the diagnosis

