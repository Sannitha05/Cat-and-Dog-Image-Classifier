# Cat-and-Dog-Image-Classifier
# Cat and Dog Image Classifier in machine learning through python 


import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_dir = "path_to_train_data"
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_data = datagen.flow_from_directory(train_dir, target_size=(150,150), batch_size=32, class_mode='binary', subset='training')
val_data = datagen.flow_from_directory(train_dir, target_size=(150,150), batch_size=32, class_mode='binary', subset='validation')

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train_data, validation_data=val_data, epochs=5)
model.save("cat_dog_classifier.h5")

