import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_curve, auc,roc_curve, recall_score, classification_report, f1_score, precision_recall_fscore_support, r2_score, ConfusionMatrixDisplay

# This file contains all helper functions to help in the data-preprocessing and result generation


# helper function to compute mean of min-max range
# apply it a column where values are given in strings

def salary_mean(val):
    ''' Computes the arithmetic mean of a min-max range'''

    if type(val) == float:
        return val
    else:
        if len(val) > 1:
            mean = (float(val[0]) + float(val[1]))/2
            return mean
        return float(val[0])


# helper function to plot the capped distribution of a numeric variable

def iqr_hist(df, col_name):
    '''Plots values within 1.5 times of the Interquartile range for a given dataframe and column.
    This function ignores NANs by default.'''

    upper, lower = df[col_name].quantile([0.75, 0.25])
    iqr = upper - lower

    print('The 25th quartile is', lower, 'and the 75th quartile is', upper)
    print('The IQR is', iqr)
    print('The arithmetic mean is', df[col_name].mean())

    df[col_name][df[col_name].between(lower - (1.5 * iqr), upper + (1.5 * iqr))].hist(color = 'black')


# helper function to compute and print accuracy, and classification scores 

def classification_performance(valid_Y, predict_Y):
    cm = confusion_matrix(valid_Y, predict_Y)
    print("Confusion matrix:")
    print(cm)

    tn = cm[0][0]
    fp = cm[0][1]
    fn = cm[1][0]
    tp = cm[1][1]

    print("\nTotal number of true positives", tp)
    print("Total number of false negatives",fn)
    print("Total number of false positives",fp)
    print("Total number of true negatives",tn)

    acc = float(tp + tn)/(tp + tn + fp + fn)

    print('\nClassifier Accuracy: %.4f%%' % (acc * 100))

    tpr = float(tp)/(tp + fn)

    print('True Positive Rate (TPR/Recall/Sensitivity): %.4f%%' % (tpr * 100))

    specificity = float (tn)/(tn + fp)

    print ("True Negative Rate (TNR/Specificity/selectivity): %.4f%%" % (specificity * 100)) 

    fpr = float(fp)/(fp + tn)
    print("False Positive Rate (FPR): %.4f%%" % (fpr * 100))

    fnr = fn/ (fn + tp)
    print("False Negative Rate (FNR): %.4f%%" % (fnr * 100))

    precision=float(tp)/(tp + fp)
    print("Precision/Positive Predictive value: %.4f%%" % (precision * 100))

    fscore = 2 * ((precision * tpr)/(precision + tpr))
    print("F1-Score: %.4f%%" %(fscore * 100))

    ConfusionMatrixDisplay.from_predictions(valid_Y, predict_Y)




def make_result_df(data, predictors, estimator):
    pred = estimator.predict(predictors) 
    proba = estimator.predict_proba(predictors) 
    result = pd.DataFrame({'actual': data, 
                            'p(0)': [p[0] for p in proba], 
                            'p(1)': [p[1] for p in proba], 
                            'predicted': pred}) 

    result.sort_values(by = 'p(1)', ascending = False, inplace = True)
    return result

