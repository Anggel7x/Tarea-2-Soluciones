from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, AveragePooling2D, Activation


def define_model():
    model = Sequential()
    model.add(Conv2D(6, 5, activation="tanh", input_shape=(28, 28, 1)))
    model.add(AveragePooling2D(2))
    model.add(Activation("sigmoid"))
    model.add(Conv2D(16, 5, activation="tanh"))
    model.add(Activation("sigmoid"))
    model.add(Conv2D(120, 5, activation="tanh"))
    model.add(Flatten())
    model.add(Dense(84, activation="tanh"))
    model.add(Dense(10, activation="softmax"))
    return model


lenet_model = define_model()
