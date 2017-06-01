import os
import numpy as np
from config import config
from model import PhoneModel
from sklearn.model_selection import train_test_split
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

frames = np.load("./data/data_in.npy")
transcript = np.load("./data/data_trans.npy")

nb_char = 30  # 29 + <blank>

id_to_char = []
with open("./char_set.txt", 'r') as f:
    lines = f.readlines()
    for line in lines:
        id_to_char.append(line.split()[-1])

model = PhoneModel(config, nb_char, id_to_char)
model.build()

X_train, X_val, y_train, y_val = train_test_split(frames, transcript, test_size=0.10, random_state=42)
train_data = list(zip(X_train, y_train))
val_data = list(zip(X_val, y_val))
model.train(train_data, val_data)