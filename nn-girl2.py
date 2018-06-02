from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Activation, Dropout, Dense
from keras.utils import np_utils
import numpy as np

#変数の宣言
classes = 3 #いくつに分類するか
data_size = 75 * 75 * 3 #縦75 x 横75 x 3原色

#データを学習、モデルを評価
def main():
  #読み込み
  data = np.load("./photo-min2.npz")
  X = data["X"] #画像データ
  y = data["y"] #ラベル
  #テストデータの読み込み
  data = np.load("./photo-test.npz")
  X_test = data["X"]
  y_test = data["y"]
  #高次元行列を２次元へ
  X = np.reshape(X, (-1, data_size))
  #訓練とテストデータ
  X_test = np.reshape(X_test, (-1, data_size))
  print()
  #モデル訓練して評価
  model = train(X, y)
  model_eval(model, X_test, y_test)

#モデルを構築しデータを学習する
def train(X, y):
  model = Sequential()
  model.add(Dense(units=64, input_dim=(data_size)))
  model.add(Activation('relu'))
  model.add(Dense(units=classes))
  model.add(Activation('softmax'))
  model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
  model.fit(X, y, epochs=30)
  model.save_weights("girl.hdf5")
  return model

#モデル評価
def model_eval(model, X_test, y_test):
  score = model.evaluate(X_test, y_test)
  print("loss:", score[0]*100, "%")
  print("accuracy:", score[1]*100, "%")

if __name__=="__main__":
  main()