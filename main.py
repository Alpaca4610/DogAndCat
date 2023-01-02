import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
from PIL import Image
import os
import matplotlib.image as mpimg
import warnings

warnings.filterwarnings('ignore')

from tensorflow import keras
from keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.preprocessing import image_dataset_from_directory
import keras.utils as image

# path = '/Users/alpaca/Downloads/dog-vs-cat-classification/train/train'
# classes = os.listdir(path)
# print(classes)
#
# fig = plt.gcf()
# fig.set_size_inches(16, 16)
#
# cat_dir = os.path.join('/Users/alpaca/Downloads/dog-vs-cat-classification/train/train/Cat')
# dog_dir = os.path.join('/Users/alpaca/Downloads/dog-vs-cat-classification/train/train/Dog')
# cat_names = os.listdir(cat_dir)
# dog_names = os.listdir(dog_dir)
#
# pic_index = 210
#
# cat_images = [os.path.join(cat_dir, fname)
#               for fname in cat_names[pic_index - 8:pic_index]]
# dog_images = [os.path.join(dog_dir, fname)
#               for fname in dog_names[pic_index - 8:pic_index]]
#
# for i, img_path in enumerate(cat_images + dog_images):
#     sp = plt.subplot(4, 4, i + 1)
#     sp.axis('Off')
#
#     img = mpimg.imread(img_path)
#     plt.imshow(img)
#
# plt.show()

base_dir = '/Users/alpaca/Downloads/dog-vs-cat-classification/train/train'
test_dir = '/Users/alpaca/Downloads/kagglecatsanddogs_5340/PetImages'

num_skipped = 0

for folder_name in ("Cat", "Dog"):
    folder_path = os.path.join(base_dir, folder_name)
    for fname in os.listdir(folder_path):
        fpath = os.path.join(folder_path, fname)
        if fname.endswith('.jpg'):
            try:
                img = Image.open(fpath)  # open the image file
                exif_data = img._getexif()
                img.verify()  # verify that it is, in fact an image
            except:
                num_skipped += 1
                # Delete corrupted image
                os.remove(fpath)
print("PIL deleted %d images" % num_skipped)

# Create datasets
train_datagen = image_dataset_from_directory(base_dir,
                                             image_size=(200, 200),
                                             subset='training',
                                             seed=1,
                                             validation_split=0.1,
                                             batch_size=32)

test_datagen = image_dataset_from_directory(test_dir,
                                            image_size=(200, 200),
                                            subset='validation',
                                            seed=1,
                                            validation_split=0.1,
                                            batch_size=32)

model = tf.keras.models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(200, 200, 3)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.1),
    layers.BatchNormalization(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.2),
    layers.BatchNormalization(),
    layers.Dense(1, activation='sigmoid')
])

model.summary()

keras.utils.plot_model(
    model,
    show_shapes=True,
    show_dtype=True,
    show_layer_activations=True
)

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

history = model.fit(train_datagen,
                    epochs=50,
                    validation_data=test_datagen)

history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss', 'val_loss']].plot()
history_df.loc[:, ['accuracy', 'val_accuracy']].plot()
plt.show()

model.save('/Users/alpaca/Downloads/kagglecatsanddogs_5340/epochs50.h5')
print("训练完成，模型已保存")
