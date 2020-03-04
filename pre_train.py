# coding:utf-8

import sys
import gflags
import keras
import matplotlib.pyplot as plt

### parameters ###
# ===============================================
FLAGS = gflags.FLAGS

# data loading parameters
gflags.DEFINE_string('train_data_dir', './inputs/train/',
                     'Directory of the training data.')
gflags.DEFINE_string('dev_data_dir', './inputs/dev/',
                     'Directory of the dev data.')
# gflags.DEFINE_float('dev_sample_percentage', 0.02, 'Percentage of the training data to user for validation (dev set).')

# model parameters
gflags.DEFINE_integer('img_height', 224,
                      'The height of the image for training (default: 227).')
gflags.DEFINE_integer('img_width', 224,
                      'The width of the image for training (default: 227).')
gflags.DEFINE_integer(
    'img_channels', 3,
    'The number of channels of the image for training (default: 3).')
gflags.DEFINE_float('dropout_keep_prob', 0.7,
                    'Dropout keep probability (default: 0.7).')

# training parameters
gflags.DEFINE_float('learning_rate', 0.001, 'Learning rate for training.')
gflags.DEFINE_integer('batch_size', 32, 'The batch size for each train step.')
gflags.DEFINE_integer('num_epochs', 10,
                      'Number of training epochs (default: 200).')

FLAGS(sys.argv)
# show parameters
print('\nPARAMETERS:')
print('================================')
for attr, value in FLAGS.flag_values_dict().items():
    print('{0}: {1}'.format(attr.lower(), value))
print('================================\n\n')

### use the pre-trained model
# create the base pre-trained model
base_model = keras.applications.VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(FLAGS.img_height, FLAGS.img_width, FLAGS.img_channels))

# add a global spatial average pooling layer
add_model = keras.Sequential(name='additional_layers')
add_model.add(keras.layers.Flatten(input_shape=base_model.output_shape[1:]))
add_model.add(keras.layers.Dense(128, activation='relu'))
add_model.add(keras.layers.Dense(2, activation='softmax'))

model = keras.models.Model(
    inputs=base_model.input, outputs=add_model(base_model.output))

# freeze all VGG16 layers
for layer in model.layers[:-1]:
    layer.trainable = False

model.compile(
    loss='categorical_crossentropy',
    optimizer=keras.optimizers.SGD(lr=FLAGS.learning_rate, momentum=0.9),
    metrics=['accuracy'])

model.summary()

train_datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
validation_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    directory=FLAGS.train_data_dir,
    target_size=(FLAGS.img_height, FLAGS.img_width),
    batch_size=FLAGS.batch_size,
    class_mode='categorical',
    seed=272)
validation_generator = validation_datagen.flow_from_directory(
    directory=FLAGS.dev_data_dir,
    target_size=(FLAGS.img_height, FLAGS.img_width),
    batch_size=FLAGS.batch_size,
    class_mode='categorical')

# train the model on the new data for a few epochs
history = model.fit_generator(
    # data_helper.generate_arrays_from_file(
    #     batch_size=FLAGS.batch_size, img_path_list=x_path_train, label_list=y_train, img_height=224, img_width=224),
    train_generator,
    steps_per_epoch=train_generator.n // FLAGS.batch_size,
    epochs=FLAGS.num_epochs,
    validation_data=validation_generator,
    verbose=1,
    callbacks=[
        keras.callbacks.ModelCheckpoint(
            './log/VGG16-transfer-learning.model',
            monitor='val_loss',
            save_best_only=True,
            verbose=1)
    ])

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

print("Training loss: {:.2f} / Validation loss: {:.2f}".format(
    history.history['loss'][-1], history.history['val_loss'][-1]))
print("Training accuracy: {:.2f}% / Validation accuracy: {:.2f}%".format(
    100 * history.history['acc'][-1], 100 * history.history['val_acc'][-1]))
