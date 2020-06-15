import numpy as np
import os
import nibabel
from scipy import ndimage
import keras
# from tensorflow import keras
# import tensorflow.python.keras as keras
# import tensorflow.keras as keras

shape = [256, 256, 256]
#shape = [64, 64, 64]
def generator(path, batch_size):
    files = os.listdir(path)
    N = len(files)
    while True:
        np.random.shuffle(files)
        for i in range(0, N, batch_size):
            if i+batch_size >= N:
                break
            imgs = []
            labels = []
            for j in range(i, i+batch_size):
                name = files[j]
                tag = name.split('_')[0]
                if tag == 'AD':
                    label = [1, 0, 0, 0]
                    label = 0
                elif tag == 'CN':
                    label = [0, 1, 0, 0]
                    label = 1
               
                elif tag == 'MCI':
                    label = [0, 0, 1, 0]
                    label = 2
                elif tag == 'SMC':
                    label = [0, 0, 0, 1]
                    label = 3
                else:
                    break
                labels.append(label)
                img = nibabel.load(os.path.join(path, name)).get_fdata()
                if len(img.shape) == 4:
                    img = img[:, :, :, 0]
                img = ndimage.zoom(img, zoom=[shape[0]/img.shape[0], shape[1]/img.shape[1], shape[2]/img.shape[2]])
                imgs.append(img)
            imgs = np.array(imgs, dtype=np.float32)
            imgs = imgs[:, :, :, :, np.newaxis]
            labels = np.array(labels, dtype=np.float32)
            yield imgs, labels
            # return imgs, labels

def CNNModel():

    input = keras.layers.Input(shape=shape+[1])
    x = keras.layers.Conv3D(32, (5, 5, 5), padding='SAME', strides=[8, 8, 8])(input)
    x = keras.layers.Conv3D(64, (5, 5, 5), activation='relu', padding='SAME', strides=[2, 2, 2])(x)
    shortcut = x
    
    x = keras.layers.Conv3D(128, (5, 5, 5), activation='relu', padding='SAME')(x)
    x = keras.layers.Conv3D(64, (5, 5, 5), activation='relu', padding='SAME')(x)
    x = keras.layers.Add()([shortcut, x])
    

    x = keras.layers.MaxPooling3D(pool_size=[2, 2, 2])(x)
    x = keras.layers.Conv3D(32, (5, 5, 5), activation='relu', padding='SAME')(x)
    shortcut = x
    attention = keras.layers.Conv3D(1, (5, 5, 5), activation='sigmoid', padding='SAME')(x)
    x = keras.layers.Conv3D(64, (5, 5, 5), activation='relu', padding='SAME')(x)
    x = keras.layers.Conv3D(32, (5, 5, 5), activation='relu', padding='SAME')(x)
    x = keras.layers.Add()([shortcut, x])
    attention = keras.layers.concatenate([attention] * int(x.shape[-1]), axis=4)
    x = keras.layers.Multiply()([x, attention])
    x = keras.layers.Conv3D(16, (5, 5, 5), activation='relu', padding='SAME')(x)

    x = keras.layers.MaxPooling3D(pool_size=[2, 2, 2])(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(64, activation='relu')(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(4, activation='softmax')(x)

    model = keras.Model(input, x)
    return model


if __name__ == '__main__':
    status = 'Training'

    model = CNNModel()
    model.summary()
    if status == 'Training':
        trainingGenerator = generator('Data/Training-4', batch_size=1)
        validationGenerator = generator('Data/Validation-4', batch_size=1)
        optimizer = keras.optimizers.Adam(learning_rate=0.005)
        model.compile(optimizer=optimizer,
                      # loss=keras.losses.categorical_crossentropy,
                      loss=keras.losses.sparse_categorical_crossentropy,
                      metrics=['accuracy'])
        tensorboard = keras.callbacks.TensorBoard(log_dir=os.path.join('record3'))
        checkpoint = keras.callbacks.ModelCheckpoint(filepath=os.path.join('record3','Test2lastattention.hdf5'), save_best_only=True,save_weights_only=True)
        model.fit_generator(
            generator=trainingGenerator,
            steps_per_epoch=50,
            epochs=100,
            validation_data=validationGenerator,
            validation_steps=4,
            callbacks=[tensorboard, checkpoint]
        )