from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json

dataset = loadtxt("pima-indians-diabetes.csv", delimiter=',')
x = dataset[:,0:8]
y = dataset[:,8]
print(x)
print(y)

model = Sequential()
model.add(Dense(180,input_dim=8,activation='relu'))
model.add(Dense(90,activation='relu'))
model.add(Dense(1,activation='sigmoid')) #sigmoid function to get the probability distribution
model.summary

model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
model.fit(x,y,epochs=1500,batch_size=80)
_,accuracy = model.evaluate(x,y) #overall accuracy
print('Accuracy: %.2f' % (accuracy*100))

model_json = model.to_json()
with open("model.json","w") as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")
print("Saved model to disk")
