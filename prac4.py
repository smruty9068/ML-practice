import numpy as np
import pandas as pd
from tensorflow.keras.layers import Dense, MaxPooling2D, Conv2D, Flatten
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from sklearn.metrics import precision_score

(X_train,y_train),(X_test,y_test)=fashion_mnist.load_data()

X_train=X_train.astype('float32')/255.0
X_test=X_test.astype('float32')/255.0

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

models=Sequential([
    Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)),
    MaxPooling2D((2,2)),
    Conv2D(64,(3,3),activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(64,activation='relu'),
    Dense(10,activation='softmax')
    ])

models.summary()

models.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

history=models.fit(X_train,y_train,validation_split=0.2,epochs=5,batch_size=10,verbose=2)
test_loss,test_accuracy=models.evaluate(X_test,y_test)

prediction=models.predict(X_test)

y_pred=np.argmax(prediction,axis=1)

precision=precision_score(y_test,y_pred,average='weighted',zero_division=0)

print(test_loss)
print(test_accuracy)
print(y_pred[0])#predicted class
print(y_test[0])#actual class
print(precision)