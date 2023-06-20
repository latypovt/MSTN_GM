import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.preprocessing import LabelBinarizer



def plot_confusion_matrix(conf_matrix, class_names, savefig = 'confusion_matrix.png'):

    plt.figure(figsize=(8,6), dpi=100)
    sns.set(font_scale = 1.1)

    ax = sns.heatmap(conf_matrix, annot=True, cmap="Blues")

    # set x-axis label and ticks. 
    ax.set_xlabel("Predicted duration", fontsize=14, labelpad=20)
    ax.xaxis.set_ticklabels(class_names, rotation=45)

    # set y-axis label and ticks
    ax.set_ylabel("Actual duration", fontsize=14, labelpad=20)
    ax.yaxis.set_ticklabels(class_names,rotation=45)

    # set plot title
    ax.set_title("Confusion Matrix for the TN Response Model", fontsize=14, pad=20)
    if savefig:
        plt.savefig(savefig, dpi=300, bbox_inches='tight')



def plot_auc(fpr, tpr, roc_auc, class_names, savefig = 'roc.png'):
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    if savefig:
        plt.savefig(savefig, dpi=300, bbox_inches='tight')

def calculate_tpr_fpr(y_real, y_pred):
    '''
    Calculates the True Positive Rate (tpr) and the True Negative Rate (fpr) based on real and predicted observations
    
    Args:
        y_real: The list or series with the real classes
        y_pred: The list or series with the predicted classes
        
    Returns:
        tpr: The True Positive Rate of the classifier
        fpr: The False Positive Rate of the classifier
    '''
    
    # Calculates the confusion matrix and recover each element
    cm = confusion_matrix(y_real, y_pred)
    TN = cm[0, 0]
    FP = cm[0, 1]
    FN = cm[1, 0]
    TP = cm[1, 1]
    
    # Calculates tpr and fpr
    tpr =  TP/(TP + FN) # sensitivity - true positive rate
    fpr = 1 - TN/(TN+FP) # 1-specificity - false positive rate
    
    return tpr, fpr

def get_all_roc_coordinates(y_real, y_proba):
    '''
    Calculates all the ROC Curve coordinates (tpr and fpr) by considering each point as a threshold for the predicion of the class.
    
    Args:
        y_real: The list or series with the real classes.
        y_proba: The array with the probabilities for each class, obtained by using the `.predict_proba()` method.
        
    Returns:
        tpr_list: The list of TPRs representing each threshold.
        fpr_list: The list of FPRs representing each threshold.
    '''
    tpr_list = [0]
    fpr_list = [0]
    for i in range(len(y_proba)):
        threshold = y_proba[i]
        y_pred = y_proba >= threshold
        tpr, fpr = calculate_tpr_fpr(y_real, y_pred)
        tpr_list.append(tpr)
        fpr_list.append(fpr)
    return tpr_list, fpr_list




# create a function to calculate the AUC for one-vs-rest classification and plot it using seaborn
def plot_roc_auc(y_test, y_pred, y_proba, class_names, colors, savefig = 'roc.png'):
    # convert the test labels from integers to dummy variables (i.e. one hot encoded)
    label_binarizer = LabelBinarizer()
    y_oh_test = label_binarizer.fit_transform(y_test)
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    # plot the ROC curve for each label
    for i in range(len(class_names)):
    # Gets the class
        fpr[i], tpr[i], _ = roc_curve(y_oh_test[:,i], y_proba[:,i], drop_intermediate=False)
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # compute micro-average ROC curve and ROC area
    
    fpr_grid = np.linspace(0.0, 1.0, 1000)

    # Interpolate all ROC curves at these points
    mean_tpr = np.zeros_like(fpr_grid)

    for i in range(len(class_names)):
        mean_tpr += np.interp(fpr_grid, fpr[i], tpr[i])  # linear interpolation

    # Average it and compute AUC
    mean_tpr /= len(class_names)

    fpr["micro"] = fpr_grid
    tpr["micro"] = mean_tpr
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    ax, fig = plt.subplots(figsize=(10,8))
    for i in range(len(class_names)):
        sns.lineplot(x=fpr[i], y=tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'
                    ''.format(class_names[i], roc_auc[i]), estimator=None, color=colors[i], linestyle='-', linewidth=2, alpha=0.8)

    sns.lineplot(x=[0, 1], y=[0, 1], color='grey', lw=2, linestyle='--', alpha=0.6)
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    #plot micro-average ROC curve and ROC area
    sns.lineplot(x=fpr["micro"], y=tpr["micro"], label='Average ROC curve (area = {0:0.2f})'
                    ''.format(roc_auc["micro"]), color='navy', linestyle='-', linewidth=2, alpha=0.7)
    if savefig:
        plt.savefig(savefig, dpi=300, bbox_inches='tight')
        



    
