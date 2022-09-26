# TensorFlow Keras
import tensorflow as tf
from tensorflow import keras
from keras import layers, activations

# get dataset (TrainSet, TestSet) = Data
fashion_mnist = keras.datasets.fashion_mnist

# label/classification [0: T-shirt/top, 1: 	Trouser] etc.
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# scale the values to 0.0 to 1.0
train_images = train_images / 255.0
test_images = test_images / 255.0

# reshape for model input
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)

# adjust dynamic parameter for testing

# activation function: activations.sigmoid, activations.softmax etc.
fnActivation = activations.relu

# epochs
epochs = 5
# learning rate
lr = 0.0001

# input-shape - image size
# kernel size - weights

model = tf.keras.Sequential()
Conv2D = layers.Conv2D
BatchNormalization = layers.BatchNormalization

model.add(Conv2D(input_shape=(28, 28, 1), filters=32, kernel_size=(5, 5), activation=fnActivation, name='Conv1', use_bias=True))
model.add(BatchNormalization())
# model.add(layers.MaxPool2D(pool_size=(2,2), strides=2))
model.add(layers.Dropout(0.2))

# model.add(Conv2D(input_shape=(28,28,1), filters=64, kernel_size=(5, 5),activation=fnActivation, name='Conv2', use_bias=True))
# model.add(BatchNormalization())
# model.add(layers.MaxPool2D(pool_size=(2,2), strides=2))
# model.add(layers.Dropout(0.4))

# classify output with Flatten for classification
model.add(layers.Flatten())
# 10
model.add(layers.Dense(10, name='Dense'))
# model.add(BatchNormalization())
# model.add(layers.Activation(fnActivation))
# model.add(layers.Dropout(0.2))

# compile
model.compile(optimizer=keras.optimizers.Adam(lr=lr),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=[keras.metrics.SparseCategoricalAccuracy()])
# train
model.fit(train_images, train_labels, epochs=epochs, shuffle=True)

# result
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('\nTest accuracy: {}'.format(test_acc))
print('\nTest loss: {}'.format(test_loss))