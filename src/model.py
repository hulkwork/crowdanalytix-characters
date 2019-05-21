from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
#create model

def  simple_cnn(target_size, shape_target = (224,224,3)):
    model = Sequential()
    #add model layers
    model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=shape_target))
    model.add(Conv2D(32, kernel_size=3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(target_size, activation='softmax'))
    return model