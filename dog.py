import tensorflow as tf
tf.__version__

train_path = "D:/projects/dogs/train"
test_path = "D:/projects/dogs/test"
valid_path = "D:/projects/dogs/valid"

import os
import random as r
files = []
label = []
for dirname, _, filenames in os.walk(train_path):
    print(dirname)  # Add this line to print the value of dirname
    for filename in filenames:
        files.append(os.path.join(dirname, filename))
        if len(dirname.split("\\")) >= 5 and dirname.split("\\")[4] not in label:
            label.append(dirname.split("\\")[4])



from keras.preprocessing.image import ImageDataGenerator
image_generator = ImageDataGenerator(rescale=1/255, validation_split=0.2)

train_dataset = image_generator.flow_from_directory(batch_size=32,
                                                 directory=train_path,
                                                 shuffle=True,
                                                 target_size=(224,224),
                                                 subset="training",
                                                 class_mode='categorical')

validation_dataset = image_generator.flow_from_directory(batch_size=32,
                                                 directory=valid_path,
                                                 shuffle=True,
                                                 target_size=(224,224),
                                                 subset="validation",
                                                 class_mode='categorical')

IMG_SHAPE = (224,224,3)
base_model = tf.keras.applications.InceptionV3(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')

base_model.trainable = False
model = tf.keras.Sequential([
  base_model,
  #tf.keras.layers.Conv2D(128, 3, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.GlobalAveragePooling2D(),
  tf.keras.layers.Dense(70, activation='softmax')
])
model._name = "Booge_man"
model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001), #Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

hist = model.fit_generator(
    train_dataset,
    epochs=10,
    validation_data = validation_dataset
)

model.save("dog_v2.h5")

import numpy as np
#testing model accuracy with predictions
def pred_funs(address):
    image = tf.keras.preprocessing.image.load_img(address, target_size=(224, 224))
    img = np.array(image)
    img = img / 255.0
    img = img.reshape(1,224,224,3)
    result = np.argmax(model.predict(img))
    return label[result]
for dirname, _, filenames in os.walk(test_path):
    for filename in filenames:
        files.append(os.path.join(dirname, filename))
        if dirname.split("/")[4] not in label:
            label.append(dirname.split("/")[4])
r.shuffle(files)
label.sort()

model.save("trained_model_inception.h5")
print("Model saved successfully!")