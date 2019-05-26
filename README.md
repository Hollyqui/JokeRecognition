## Goal and Initial Idea

Upon reading previous papers it came to our notice that previous humour recognition papers are based on a comparison between jokes and news headlines. The results attained with those papers are questionable since news headlines do not represent spoken everyday language (such as used in most jokes). Therefore, the identification task becomes nearly trivial (with a fairly simple network we were able to achieve a prediction accuracy of ca. 93%). To combat this 'systematic error' this repository is dedicated to creating/testing a classifier on a comparison between jokes and quotes. This is a much more difficult task as both are spoken language and both contain few rare words.  

## Dataset information

Yet to come

## Network

A convolutional network was created using Keras and Tensorflow. It consists of 7 layers as shown in the source code below.

```markdown

# Network Architecture:

model = keras.Sequential([
   keras.layers.Conv1D(filters,
                       kernel_size=(3),
                       activation='relu'),
   keras.layers.MaxPool1D((2)),
   keras.layers.Dropout(0.25),
   keras.layers.Flatten(),
   keras.layers.Dense(128, activation='relu'),
   keras.layers.Dropout(0.5),
   keras.layers.Dense(2, activation='softmax')
])

```
The network is trained using the Adam optimizer and the Categorical Cross Entropy loss function

## Results

All the graphs and results were computed using test data and should therefore not show inflated accuracy because of overtraining.

![Figure 1: roc curve comparison jokes & headlines vs jokes & quotes](https://github.com/Hollyqui/JokeRecognition/blob/master/roc_curve_joke.jpeg)

Figure 1 shows how the performance of the network differs depending on what dataset is used for the negatives. It clearly shows that the classification between jokes and headlines is a much easier task, featuring an AUC (*A*rea *U*nder the *C*urve) of 0.991, whereas the classification between jokes and quotes only yields an AUC of 0.764. This proves the initial assumption, that the comparison in previous papers contains a systematic error through the choice of datasets. The high accuracies found by previous papers (ca. 83%) is probably achieved because news headlines have very obvious identifying features. 

![Figure 2: histogram jokes vs headlines](https://github.com/Hollyqui/JokeRecognition/blob/master/histogram_joke_headline.png)

The findings of Figure 1 are also supported by Figure 2 and Figure 3. Figure 2 (jokes vs headlines) shows an nearly ideal histogram with barely any overlap in between the classes. Figure 3 (jokes vs quotes) shows a significant overlap of the classes, indicating some false predictions. Additionally, the typical peaks at 0 and 1 are far less pronounced, indicating a more difficult classification task.

![Figure 3: histogram jokes vs quotes](https://github.com/Hollyqui/JokeRecognition/blob/master/histogram_joke_quote.png)



### Contact

This project is developed by Szymon Fonau [@Sarkosos](https://github.com/Sarkosos) and Felix Quinque [@Hollyqui](https://github.com/Hollyqui) as a project for [Maastricht University](maastrichtuniversity.nl)(supervised by Jerry Spanakis). If you have any questions about the project you can contact us via:
s.fonau@student.maastrichtuniversity.nl
f.quinque@student.maastrichtuniversity.nl
