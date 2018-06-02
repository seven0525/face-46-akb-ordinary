from keras.models import Sequential
from keras.layers import Activation, Dense
from PIL import Image
import numpy as np, sys

classes = 3
photo_size = 75
data_size = photo_size * photo_size * 3
labels = ["48系統", "46系統", "普通の人"]

def build_model():
  model = Sequential()
  model.add(Dense(units=64, input_dim=(data_size)))
  model.add(Activation('relu'))
  model.add(Dense(units=classes))
  model.add(Activation('softmax'))
  model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='sgd',
    metrics=['accuracy'])
  model.load_weights('girl.hdf5')
  return model

def check(model, fname):
  img = Image.open(fname)
  img = img.convert('RGB')
  img = img.resize((photo_size, photo_size))
  data = np.asarray(img).reshape((-1, data_size)) / 256

  res =model.predict([data])[0]
  y = res.argmax()
  per = int(res[y] * 100)
  print("{0} ({1} %)".format(labels[y], per))

if len(sys.argv) <= 1:
  print("check.py ファイル名")
  quit()

model = build_model()
check(model, sys.argv[1])