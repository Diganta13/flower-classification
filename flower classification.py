hyperparameter = {
    'layer_1': [256, 512],
    'activation': ['relu', 'elu'],
    'dropout': [0.5, 0.6]
}

import matplotlib.pyplot as plt
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.models import Sequential, Model
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        'put your own directory',
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        'put your own directory',
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
        'put your own directory',
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical')


#Loading model
# base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))


def build_finetune_model(base_model, dropout, num_classes, fc_layers, activation):
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = Flatten()(x)

    # New FC layer, random init
    x = Dense(fc_layers, activation=activation)(x)
    x = Dropout(dropout)(x)

    # New softmax layer
    predictions = Dense(num_classes, activation='softmax')(x)

    finetune_model = Model(inputs=base_model.input, outputs=predictions)

    return finetune_model

def create_evaluate_model(hyperparameter):


    finetune_model = build_finetune_model(base_model, dropout=hyperparameter[2], num_classes=17, fc_layers=hyperparameter[0],activation=hyperparameter[1])

    return finetune_model

np.random.seed(1)
random.seed(1)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
                              inter_op_parallelism_threads=1)

tf.set_random_seed(1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)


import  itertools

count = 0

for element in itertools.product(*hyperparameter.values()):
    print("Hyperparameters ")
    print(count)
    print(element)
    count = count+1
    finetune_model = create_evaluate_model(element)
    finetune_model.compile(Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    filepath = "weights-improvement-iteration{count}".format(count=count)+"{epoch:02d}-{val_acc:.4f}.hdf5"

    checkpoint = ModelCheckpoint(filepath, monitor="val_acc", verbose=1, mode='max', save_best_only=True)
    callbacks_list = [checkpoint]

    # finetune_model.summary()

    history = finetune_model.fit_generator(train_generator,
                                           epochs=10,
                                           steps_per_epoch=train_generator.samples // train_generator.batch_size,
                                           shuffle=True, callbacks=callbacks_list,
                                           validation_data=validation_generator,
                                           validation_steps=10)



#Evaluating the model with best accuracy
test_model = create_evaluate_model((512, 'relu', 0.6))
test_model.load_weights("weights-improvement-iteration610-0.8562.hdf5")
test_model.compile(Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
score = test_model.evaluate_generator(test_generator, 10)
print(score)