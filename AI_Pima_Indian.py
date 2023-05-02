from keras.models import Sequential
from keras.layers import Dense
import numpy
import tensorflow as tf

# seed 값 생성
seed = 0
numpy.random.seed(seed)
tf.random.set_seed(seed)

# 데이터 로드
dataset = numpy.loadtxt("../dataset/pima-indians-diabetes.csv", delimiter=",")
X = dataset[:,0:8]
Y = dataset[:,8]

model = Sequential()
model.add(Dense(20, input_dim=8, activation='swish'))
model.add(Dense(50, activation='swish'))
model.add(Dense(45, activation='swish'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

results = model.fit(X, Y, epochs = 350, batch_size = 30)

print("\n Accuracy : %.4f" % (model.evaluate(X, Y)[1]))



