import threading, os
import numpy as np
import keras
import matplotlib.pyplot as plt
from keras import callbacks, regularizers
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
import keras.backend as K

#HYPER PARAMETERS
WEIGHTS_FILE = 'weights.w'
DATASET_FILE = 'dataset.npz'
TRAIN_EPOCHS = 100
LOAD_WEIGHTS = False

cache = np.load(DATASET_FILE)
inputSet = cache['inputSet']
outputSet = cache['outputSet']
#idxToPassengerId = cache['idxToPassengerId']

print("Input shape: {}".format(inputSet.shape))
print("Output shape: {}".format(outputSet.shape))

def customLoss(yTrue, yPred):
	intermediate = K.sum(K.square((yTrue - yPred) / yTrue))
	rmpse = K.sqrt(intermediate)
	return rmpse

# Training
model = Sequential()
model.add(Dense(100, activation='relu', input_shape=[inputSet.shape[1]]))
model.add(Dropout(0.5))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['acc'])

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

print(history)

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
