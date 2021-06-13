from tensorflow.python.keras import backend as K
from keras.models import Model
from keras.layers import Flatten, Dense, Dropout
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import ResNet50 
from datetime import datetime
import tensorflow as tf
import keras
DATASET_PATH  = 'sample_r'
IMAGE_SIZE = (224, 224)
NUM_CLASSES = 2
BATCH_SIZE = 16
FREEZE_LAYERS = 10
NUM_EPOCHS = 5
WEIGHTS_FINAL = 'model-resnet50_r.h5'

train_datagen = ImageDataGenerator(rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   channel_shift_range=10,
                                   horizontal_flip=True,
                                   fill_mode='nearest')
train_batches = train_datagen.flow_from_directory(DATASET_PATH + '/train',
                                                  target_size=IMAGE_SIZE,
                                                  interpolation='bicubic',
                                                  class_mode='categorical',
                                                  shuffle=True,
                                                  batch_size=BATCH_SIZE)

valid_datagen = ImageDataGenerator()
valid_batches = valid_datagen.flow_from_directory(DATASET_PATH + '/valid',
                                                  target_size=IMAGE_SIZE,
                                                  interpolation='bicubic',
                                                  class_mode='categorical',
                                                  shuffle=False,
                                                  batch_size=BATCH_SIZE)

for cls, idx in train_batches.class_indices.items():
    print('Class #{} = {}'.format(idx, cls))

net = ResNet50(include_top=False, weights='imagenet', input_tensor=None,
               input_shape=(IMAGE_SIZE[0],IMAGE_SIZE[1],3))
x = net.output
x = Flatten()(x)

x = Dropout(0.5)(x)

output_layer = Dense(NUM_CLASSES, activation='softmax', name='softmax')(x)

net_final = Model(inputs=net.input, outputs=output_layer)
for layer in net_final.layers[:FREEZE_LAYERS]:
    layer.trainable = False
for layer in net_final.layers[FREEZE_LAYERS:]:
    layer.trainable = True

log_dir = './callback_r'
callbacks = [keras.callbacks.TensorBoard(log_dir=log_dir,
                                         histogram_freq=1,
                                         embeddings_freq=0,
                                         embeddings_layer_names=None,
                                         ),]

net_final.compile(optimizer=Adam(lr=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

print(net_final.summary())


net_final.fit(          train_batches,
                        steps_per_epoch = train_batches.samples // BATCH_SIZE,
                        validation_data = valid_batches,
                        validation_steps = valid_batches.samples // BATCH_SIZE,
                        epochs = NUM_EPOCHS,
                        callbacks=callbacks )
'''tf.summary.merge_all() 
file_writer = tf.summary.FileWriter(logdir,tf.get_default_graph())
file_writer.close()'''

net_final.save(WEIGHTS_FINAL)