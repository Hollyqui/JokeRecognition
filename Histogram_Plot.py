import matplotlib.pyplot as plt
import numpy as np
import pandas
import tensorflow as tf
import numpy as np
import pandas as pd

# function plots a histogram for any binary classifier; needs input of labels and network predictions
def plot_histogram(labels, predictions):
    class_var = np.array( [predictions[i][0] for i in range(len(predictions)) ] ) # list of prob_pion
    df_test = pd.DataFrame()
    df_test["Net"] = class_var
    df_test["absid"] = labels.T[0]

    crit_class1 = df_test['absid'] == 1
    crit_class2 = df_test['absid'] == 0
    df_test_class1 = df_test[crit_class1]
    df_test_class2 = df_test[crit_class2]
    #log="y" transforms the scale to be logarithmic (in the following two lines)
    df_test_class1["Net"].plot.hist(bins=50, range=(0,1), alpha=0.5, density=True, label="Headline") # log="y")
    df_test_class2["Net"].plot.hist(bins=50, range=(0,1), alpha=0.5, density=True, label="Joke") #log="y") 
    plt.legend(loc='upper right')
    plt.xlabel("ConvNet classifier")
    plt.show()

    f = plt.figure()
    f.savefig("performance.pdf", bbox_inches='tight')

labels =  np.load("labels_headlines.npy")
predictions = np.load("predictions_headlines.npy")
plot_histogram(labels, predictions)
