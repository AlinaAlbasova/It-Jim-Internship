import tensorflow as tf
import numpy as np
import json


# Importing the required Keras modules containing model and layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, Reshape, \
    Input, Concatenate, MaxPooling2D, BatchNormalization


import os

from tensorflow.python.keras.utils.np_utils import to_categorical
from keras.preprocessing import image

TRAIN_DIR = "splitted_dataset/train/"
TEST_DIR = "splitted_dataset/test/"
VAL_DIR = "splitted_dataset/val/"

train_images = [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR)]
val_images = [VAL_DIR+i for i in os.listdir(VAL_DIR)]
test_images =[TEST_DIR+i for i in os.listdir(TEST_DIR)]

classes = os.listdir("dataset/")
class_numbers = range(0,16)

from data_preparation import prep_data
from models_eval import eval

def load_model(fnm):
    import json
    from tensorflow.keras.models import model_from_json
    with open(fnm + '.txt') as infile:
        json_string = json.load(infile)
    model = model_from_json(json_string)
    model.load_weights(fnm + '.hdf5', by_name=False)
    model.compile(loss='binary_crossentropy', optimizer='Adagrad', metrics=['accuracy'])
    model.summary()
    return model


def save_model(model,fname):
    import json
    print('Saving the model')
    json_string = model.to_json()
    with open(fname + '.txt', 'w', encoding='utf-8') as f:
        json.dump(json_string, f, ensure_ascii=False)
    print('Saving the weights')
    model.save_weights(fname + '.hdf5')
    print('Done')


def train_fcnn(x_train, y_train, x_val=None, y_val=None):

    x_train /= 255.0
    x_val /= 255.0

    y_train = to_categorical(y_train)
    y_val = to_categorical(y_val)

    input_shape = (84, 84, 3) # shape of the data. Ommiting the first dimension!
    inputs = Input(shape=input_shape)
    x = Flatten()(inputs)
    x = Dense(100)(x)
    x = Dense(50)(x)
    x = Dense(30)(x)
    predictions = Dense(16, activation='softmax')(x)

    # setting up the model
    model = Model(inputs=inputs, outputs=predictions)
    model.summary()
    # opt = SGD(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

    # training
    model.fit(x_train, y_train, batch_size=32, epochs=100,validation_data=(x_val, y_val))

    return model

    # loss_and_metrics = model.evaluate(x_test, y_test, batch_size=256)
    # print('Test loss:', loss_and_metrics[0])
    # print('Test accuracy:', loss_and_metrics[1])


def train_cnn(x_train, y_train, x_val=None, y_val=None):
    x_train /= 255.0
    x_val /= 255.0

    y_train = to_categorical(y_train)
    y_val = to_categorical(y_val)

    input_shape = (84, 84, 3)
    inputs = Input(shape=input_shape)
    # Convolution layer
    x = Conv2D(10, kernel_size=(3, 3), activation='relu')(inputs)
    # Max pooling layer
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = Conv2D(10, kernel_size=(3, 3), activation='relu')(x)
    # Max pooling layer
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(100)(x)
    predictions = Dense(16, activation='softmax')(x)

    # setting up the model
    model = Model(inputs=inputs, outputs=predictions)
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

    # training!
    model.fit(x_train, y_train, batch_size=32, epochs=10, validation_data=(x_val, y_val))

    return model


def test(x_test, y_test, model):

    x_test = x_test / 255.0
    y_hat_probabilities = model.predict(x_test, verbose=0)
    yhat_classes = np.argmax(y_hat_probabilities,axis=1)

    eval(y_test, yhat_classes)


def infer(path_to, model, mode='image'):
    if mode == 'image':
        img = image.load_img(path_to, target_size=(84, 84))

        x = image.img_to_array(img)
        x = x / 255.0
        x = np.expand_dims(x, axis=0)

        pred_classes = model.predict(x)
        pred_class = np.argmax(pred_classes)
        pred_class = classes[pred_class]

        np.set_printoptions(precision=3, suppress=True)
        for i in range(16):
            print(classes[i], " - ", pred_classes[:, i])
        print("Best match for ", path_to, " is ", pred_class)

    elif mode == 'folder':
        folder_predictions = {}
        for fn in os.listdir(path_to):
            # predicting images
            path = path_to + fn
            img = image.load_img(path, target_size=(84, 84))

            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)

            pred_classes = model.predict(x)
            pred_classes = np.squeeze(pred_classes)
            pred_class = np.argmax(pred_classes)
            pred_class = classes[pred_class]

            predictions_dict = {}

            np.set_printoptions(precision=3, suppress=True)
            for i in range(16):
                predictions_dict[classes[i]] = str(pred_classes[i])
                print(classes[i], " - ", pred_classes[i])

            folder_predictions[fn] = predictions_dict

            with open('predictions.json', 'w') as fp:
                json.dump(folder_predictions, fp)

            print("Best match for ", fn, " is ", pred_class)


if __name__ == '__main__':
    X_train, Y_train = prep_data(train_images)

    X_val, Y_val = prep_data(val_images)

    X_test, Y_test = prep_data(test_images)

    # m = train_cnn(X_train, Y_train, X_val, Y_val)
    # m = train_fcnn(X_train, Y_train, X_val, Y_val)

    # save_model(m, 'cnn')
    mod = load_model('cnn')
    # m = train_cnn(X_train, Y_train, X_val, Y_val)

    test(X_test, Y_test, mod)

    path = "inf3.jpg"
    path2 = "splitted_dataset/val/"
    infer(path, mod)
    # infer(path2, mod, mode='folder')



