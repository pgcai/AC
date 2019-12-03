import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers import Dense, Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator

# from keras import backend as K

num_classes = 2
img_rows, img_cols = 160, 160
channel = 9
batch_size = 64
input_shape = (img_rows, img_cols, channel)

datagen = ImageDataGenerator()  # 也可以做数据增广
train_generator = datagen.flow_from_directory(
    './training_set',
    target_size=(224, 224),
    class_mode='categorical',
    batch_size=batch_size)

# 读验证集图片
validation_generator = datagen.flow_from_directory(
    './test_set',
    classes=['cat', 'dog'],
    target_size=(224, 224),
    class_mode='categorical',
    batch_size=batch_size)

# turn to one-hot code

# trainY = keras.utils.to_categorical(trainY,num_classes)
# testY = keras.utils.to_categorical(testY,num_classes)

# define model
model = Sequential()
model.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu', input_shape=input_shape))  # 224
model.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu', input_shape=input_shape))  # 224
model.add(MaxPooling2D(pool_size=(2, 2)))  # 110

model.add(Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu', input_shape=input_shape))  # 112
model.add(Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu', input_shape=input_shape))  # 112
model.add(MaxPooling2D(pool_size=(2, 2)))  # 110

model.add(Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu', input_shape=input_shape))  # 56
model.add(Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu', input_shape=input_shape))  # 56
model.add(Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu', input_shape=input_shape))  # 56
model.add(MaxPooling2D(pool_size=(2, 2)))  # 110

model.add(Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu', input_shape=input_shape))  # 28
model.add(Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu', input_shape=input_shape))  # 28
model.add(Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu', input_shape=input_shape))  # 28
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu', input_shape=input_shape))  # 14
model.add(Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu', input_shape=input_shape))  # 14
model.add(Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu', input_shape=input_shape))  # 14
model.add(MaxPooling2D(pool_size=(2, 2)))  # 7

model.add(GlobalAveragePooling2D())  # straightening the output
model.add(Dense(4096, activation='relu'))  # the link layera
model.add(Dense(4096, activation='relu'))  # the link layer

model.add(Dense(num_classes, activation='softmax'))

# define loss function,optimization
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True),
              metrics=['accuracy'])

# auto achieve train
# model.fit(trainX,trainY,batch = 128, epochs = 10,validation_split = 0.1,)
# validation_data = (testX,testY)
model.fit_generator(generator=train_generator, steps_per_epoch=8000 / 64, epochs=10,
                    validation_data=validation_generator, validation_steps=2025 / 64)

model.save('model_weight.h5')
# 载入模型
# model = load_model('model.h5')

# score = model.evaluate()
# print('Test loss:',score[0])
# print('Test accuracy',score[1])