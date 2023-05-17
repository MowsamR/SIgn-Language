import numpy as np
import os
from matplotlib import pyplot as plt
import time
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import load_model as load_model

if __name__ == '__main__':

	actions = np.array(['Thank_you', 'Hello', 'Good', 'Yes', 'Goodbye'])
	label_map = {label:num for num, label in enumerate(actions)}

	sequences, labels = [], []
	DATA_PATH = os.path.join('Dataset')
	for action in os.listdir(DATA_PATH):
	    if not os.path.isdir(os.path.join(DATA_PATH, action)):
	        continue
	    #print(action)
	    for sequence in os.listdir(os.path.join(DATA_PATH, action)):
	        if not os.path.isdir(os.path.join(DATA_PATH, action, sequence)):
	            continue
	        #print(f'\t{sequence}')
	        window = []
	        for frame_num in range(30):
	            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
	            window.append(res)
	        sequences.append(window)
	        labels.append(label_map[action])

	x = np.array(sequences)
	y = to_categorical(labels).astype(int)
	x_train, x_test, y_train, y_test = train_test_split(x, y , test_size = 0.05)

	log_dir = os.path.join('Logs')
	tb_callback = TensorBoard(log_dir=log_dir)

	model = Sequential()
	model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
	model.add(LSTM(128, return_sequences=True, activation='relu'))
	model.add(LSTM(64, return_sequences=False, activation='relu'))
	model.add(Dense(64, activation='relu'))
	model.add(Dense(32, activation='relu'))
	model.add(Dense(actions.shape[0], activation='softmax'))

	model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
	model.fit(x_train, y_train, epochs=400, callbacks=[tb_callback])

	model.summary()

	model.save('actions-v0.3.1-5 signs.h5')


