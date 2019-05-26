import tensorflow as tf
import keras
from sklearn.model_selection import train_test_split
import numpy as np
import nltk
from keras.utils import to_categorical
jokes = np.load("shortjokes_for_quotes_vec.npy", allow_pickle=True)
headlines = np.load('quotes_vec.npy', allow_pickle=True)
sequence_length = max(len(x) for x in jokes)
print(sequence_length)

X = np.concatenate((jokes, headlines), axis=0)
y = []

for i in range(len(jokes)):
    y.append([1, 0])
for i in range(len(headlines)):
    y.append([0, 1])
y = np.array(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5)

filters = 10

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
#compile model using accuracy to measure model performance
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train,
         y_train,
         epochs = 10,
#          callbacks=[tensorboard],
         batch_size=100,
         shuffle=True,
         validation_split=0.1
         )
predictions = model.predict(X_test)

np.save('predictions_quotes', np.array(predictions))
np.save('labels_quotes', y_test)
