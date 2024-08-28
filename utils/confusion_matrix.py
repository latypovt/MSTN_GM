import seaborn as sns
import seaborn.objects as so
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.preprocessing import LabelBinarizer



def plot_confusion_matrix(conf_matrix, class_names, savefig = 'confusion_matrix.png'):

    plt.figure(figsize=(6,4), dpi=300)
    sns.set(font_scale = 1.1)

    ax = sns.heatmap(conf_matrix, annot=True, cmap="Blues")

    # set x-axis label and ticks. 
    ax.set_xlabel("Predicted value", fontsize=14, labelpad=20)
    ax.xaxis.set_ticklabels(class_names, rotation=45)

    # set y-axis label and ticks
    ax.set_ylabel("Actual value", fontsize=14, labelpad=20)
    ax.yaxis.set_ticklabels(class_names,rotation=45)

    # set plot title
    ax.set_title("Confusion Matrix", fontsize=14, pad=20)
    if savefig:
        plt.savefig(savefig, dpi=300, bbox_inches='tight')



def plot_auc(fpr, tpr, roc_auc, class_names, savefig = 'roc.png'):
    sns.set_style("whitegrid")
    plt.figure(figsize=(6,4), dpi=300)
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
    plt.close()

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
def plot_roc_auc(y_test, y_proba, class_names, colors, savefig = 'roc.png'):

    fpr = dict()
    tpr = dict()
    roc_auc = dict()


    fpr['macro'], tpr['macro'], _ = roc_curve(y_test, y_proba[:,1], drop_intermediate=False)
    roc_auc['macro'] = auc(fpr['macro'], tpr['macro'])
    
    # compute micro-average ROC curve and ROC area
    
    fpr_grid = np.linspace(0.0, 1.0, 1000)
    mean_tpr = np.zeros_like(fpr_grid)
    mean_tpr += np.interp(fpr_grid, fpr['macro'], tpr['macro'])


    fpr["micro"] = fpr_grid
    tpr["micro"] = mean_tpr
    fpr['micro'] = np.insert(fpr['micro'], 0, 0)
    tpr['micro'] = np.insert(tpr['micro'], 0, 0)

    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    random_df = pd.DataFrame({'fpr': [0, 1], 'tpr': [0, 1]})
    df = pd.DataFrame({'fpr': fpr['micro'], 'tpr': tpr['micro']})
    df.to_csv('out/roc_auc.csv')

    p = so.Plot(
        data=df, x='fpr', y='tpr'
    ).add(
        so.Line(color='navy', linewidth=2, linestyle='-', alpha=0.6), label='Average ROC curve (area = {0:0.2f})'.format(roc_auc["micro"])
    ).add(
        so.Line(color='grey', linewidth=2, linestyle='--', alpha=0.6,), data = random_df, x = 'fpr', y = 'tpr'
    ).theme(
        sns.axes_style("whitegrid")
    ).label(
        x='False Positive Rate', y='True Positive Rate', title='Receiver operating characteristic'
    ).layout(
        size=(6,6), engine='tight'
    )

    f = plt.figure(figsize=(6,6))
    plotter = p.layout(engine='tight').theme(sns.axes_style("whitegrid")).on(f).plot()
    plotter._figure.legends[0].set_bbox_to_anchor((0.4, 0.15))

    if savefig:
        plotter.save(savefig, bbox_inches='tight', dpi=400, transparent=False)




    
