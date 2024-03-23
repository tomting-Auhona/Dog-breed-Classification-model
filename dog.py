import tensorflow as tf # Import TensorFlow for deep learning operations

# Check TensorFlow version
tf.__version__

# Define paths for training, testing, and validation data
train_path = "D:/projects/dogs/train"
test_path = "D:/projects/dogs/test"
valid_path = "D:/projects/dogs/valid"

import os # Import os for operating system related tasks (e.g., file operations, directory traversal)

files = []
label = []

# Walk through the training directory to gather file paths and labels
for dirname, _, filenames in os.walk(train_path):
    print(dirname) # Print the value of dirname
    for filename in filenames:
        files.append(os.path.join(dirname, filename))
        # Extract and store unique labels from directory names
        if len(dirname.split("\\")) >= 5 and dirname.split("\\")[4] not in label:
            label.append(dirname.split("\\")[4])


# Import required libraries for image data generation
from keras.preprocessing.image import ImageDataGenerator

# Create image data generator with rescaling and validation split
image_generator = ImageDataGenerator(rescale=1/255, validation_split=0.2)

# Generate training dataset from directory
train_dataset = image_generator.flow_from_directory(batch_size=32,
                                                 directory=train_path,
                                                 shuffle=True,
                                                 target_size=(224,224),
                                                 subset="training",
                                                 class_mode='categorical')

# Generate validation dataset from directory
validation_dataset = image_generator.flow_from_directory(batch_size=32,
                                                 directory=valid_path,
                                                 shuffle=True,
                                                 target_size=(224,224),
                                                 subset="validation",
                                                 class_mode='categorical')

# Define input shape for the model
IMG_SHAPE = (224,224,3)

# Load InceptionV3 model with pre-trained weights
base_model = tf.keras.applications.InceptionV3(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')

# Freeze base model layers to prevent them from being trained
base_model.trainable = False

# Build sequential model with InceptionV3 base, dropout, global average pooling, and dense output layer
model = tf.keras.Sequential([
  base_model,
  # tf.keras.layers.Conv2D(128, 3, activation='relu'),
  # Dropout layer to prevent overfitting
  tf.keras.layers.Dropout(0.2),
  # Global average pooling to reduce spatial dimensions
  tf.keras.layers.GlobalAveragePooling2D(),
  # Output layer with softmax activation for multi-class classification
  tf.keras.layers.Dense(70, activation='softmax')
])

# Set model name
model._name = "cool_code"

# Display model summary
model.summary()

# Compile model with Adam optimizer, categorical cross-entropy loss, and accuracy metric
model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001), #Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train model using image data generators with specified epochs and validation data
hist = model.fit_generator(
    train_dataset,
    epochs=10,
    validation_data = validation_dataset
)

# Save trained model
model.save("dog_v2.h5")