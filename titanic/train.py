import threading, os
import numpy as np
import keras
import matplotlib.pyplot as plt
from keras import callbacks, regularizers
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout

#HYPER PARAMETERS
WEIGHTS_FILE = 'weights.w'
DATASET_FILE = 'dataset.npz'
TRAIN_EPOCHS = 5000
LOAD_WEIGHTS = False

cache = np.load(DATASET_FILE)
inputSet = cache['inputSet']
outputSet = cache['outputSet']
idxToPassengerId = cache['idxToPassengerId']

print("Input shape: {}".format(inputSet.shape))
print("Output shape: {}".format(outputSet.shape))

# Training
model = Sequential()
model.add(BatchNormalization(input_shape=[inputSet.shape[1]]))
model.add(Dense(60, activation='relu',
				kernel_regularizer=regularizers.l2(0.00001),
				activity_regularizer=regularizers.l1(0.00001)))
model.add(Dropout(0.5))
model.add(Dense(30, activation='relu',
				kernel_regularizer=regularizers.l2(0.00001),
				activity_regularizer=regularizers.l1(0.00001)))
model.add(Dropout(0.2))
model.add(Dense(1, activation="sigmoid",
				kernel_regularizer=regularizers.l2(0.00001),
				activity_regularizer=regularizers.l1(0.00001)))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

if LOAD_WEIGHTS:
	if os.path.isfile(WEIGHTS_FILE):
		print("Loading weights file")
		model.load_weights(WEIGHTS_FILE)
	else:
		print("No weights file found! Not loading weights")
else:
	print("LOADING WEIGHTS DISABLED!")

checkpoint = callbacks.ModelCheckpoint(WEIGHTS_FILE,
    monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True, mode='auto', period=10)
history = model.fit(inputSet, outputSet, validation_split=0, shuffle=True,
    epochs=TRAIN_EPOCHS, callbacks=[checkpoint], batch_size = 100)

# Training History Plot
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Training History')
plt.ylabel('Values')
plt.xlabel('Epochs')
plt.ylim(ymax=1.0)
plt.legend(['Training Loss', 'Validation Loss', 'Training Accuracy', 'Validation Accuracy'])
plt.show()
