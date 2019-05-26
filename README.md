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

## Results

Yet to come

The network is trained using the Adam optimizer and the Categorical Cross Entropy loss function

### Contact

This project is developed by Szymon Fonau [@Sarkosos](https://github.com/Sarkosos) and Felix Quinque [@Hollyqui](https://github.com/Hollyqui) as a project for [Maastricht University](maastrichtuniversity.nl)(supervised by Jerry Spanakis). If you have any questions about the project you can contact us via:
s.fonau@student.maastrichtuniversity.nl
f.quinque@student.maastrichtuniversity.nl
