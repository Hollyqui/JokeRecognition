import pickle
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt
import numpy as np
import pandas


target_results_tensorflow =  np.load("labels_quotes.npy")
roc_data_tensorflow =np.load("predictions_quotes.npy")
target_results_sklearn = np.load("labels_headlines.npy")
roc_data_sklearn = np.load("predictions_headlines.npy")

for i in range(1):
    stored_labels_tensorflow = target_results_tensorflow.T[0]
    temp_labels_tensorflow = []
    temp_predictions_tensorflow = roc_data_tensorflow.T[i] #(iterates through the columns)

    stored_labels_sklearn = target_results_sklearn.T[0]
    temp_labels_sklearn = []
    temp_predictions_sklearn = roc_data_sklearn.T[i]  # (iterates through the columns)
    for j in range(len(target_results_tensorflow)):
        if stored_labels_tensorflow[j] == i:
            temp_labels_tensorflow.append(1)
        else:
            temp_labels_tensorflow.append(0)
    for k in range(len(target_results_sklearn)):
        if stored_labels_sklearn[k] == i:
            temp_labels_sklearn.append(1)
        else:
            temp_labels_sklearn.append(0)
    if i == 2:
        tpr_sklearn, fpr_sklearn, thresholds_sklearn = roc_curve(temp_labels_sklearn, temp_predictions_sklearn)
    else:
        tpr_sklearn, fpr_sklearn,  thresholds_sklearn = roc_curve(temp_labels_sklearn, temp_predictions_sklearn)
    auc_sklearn = auc(fpr_sklearn, tpr_sklearn)
    tpr_keras, fpr_keras, thresholds_keras = roc_curve(temp_labels_tensorflow, temp_predictions_tensorflow)
    auc_keras = auc(fpr_keras, tpr_keras)
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_keras, tpr_keras, label='Quotes (area = {:.3f})'.format(auc_keras))
    plt.plot(fpr_sklearn, tpr_sklearn, label='Headlines (area = {:.3f})'.format(auc_sklearn))

plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()

k= 0
confusion_matrix = []
for i in range(2):
    confusion_matrix.append([])
    for j in range(2):
        confusion_matrix[i].append(0)


for i in roc_data_tensorflow:
   predictions_quotes = np.argmax(roc_data_tensorflow[k])
   real_result = target_results_tensorflow[k][0]
   confusion_matrix[predictions_quotes][real_result]= confusion_matrix[predictions_quotes][real_result]+1
   k = k + 1

confusion_matrix = np.array(confusion_matrix)
confusion_matrix = confusion_matrix[::-1]
print('quotes: \n',confusion_matrix)
k= 0
confusion_matrix = []
for i in range(2):
    confusion_matrix.append([])
    for j in range(2):
        confusion_matrix[i].append(0)
for i in roc_data_sklearn:
   predictions_quotes = np.argmax(roc_data_sklearn[k])
   real_result = target_results_sklearn[k][0]
   confusion_matrix[predictions_quotes][real_result]= confusion_matrix[predictions_quotes][real_result]+1
   k = k + 1
confusion_matrix = np.array(confusion_matrix)
confusion_matrix = confusion_matrix[::-1]
print('headlines: \n', confusion_matrix)