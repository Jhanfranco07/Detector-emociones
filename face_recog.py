import kagglehub

# Descargar datasets
path_fer = kagglehub.dataset_download("msambare/fer2013")
path_raf = kagglehub.dataset_download("shuvoalok/raf-db-dataset")

print("FER2013 en:", path_fer)
print("RAF-DB en:", path_raf)

import tensorflow as tf
print("Dispositivos disponibles:", tf.config.list_physical_devices())

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from tqdm import tqdm

from tensorflow import keras
from keras import Input
from keras.utils import to_categorical
from keras.preprocessing.image import load_img
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import LabelEncoder
from keras.callbacks import ReduceLROnPlateau

# Directorios de FER2013 y RAF-DB
FER_TRAIN_DIR = '/kaggle/input/fer2013/train'
FER_TEST_DIR = '/kaggle/input/fer2013/test'
RAF_TRAIN_DIR = '/kaggle/input/raf-db-dataset/DATASET/train'
RAF_TEST_DIR = '/kaggle/input/raf-db-dataset/DATASET/test'

# Mapeo RAF-DB de número a nombre de emoción (importante)
raf_emotion_map = {
    '1': 'surprise',
    '2': 'fear',
    '3': 'disgust',
    '4': 'happy',
    '5': 'sad',
    '6': 'angry',
    '7': 'neutral'
}

# -------- Cargar FER2013 --------
def create_dataframe_fer(dir):
    image_paths = []
    labels = []
    for label in os.listdir(dir):
        for image_name in os.listdir(os.path.join(dir, label)):
            image_paths.append(os.path.join(dir, label, image_name))
            labels.append(label)
    return image_paths, labels

fer_train = pd.DataFrame()
fer_train['image'], fer_train['label'] = create_dataframe_fer(FER_TRAIN_DIR)
fer_test = pd.DataFrame()
fer_test['image'], fer_test['label'] = create_dataframe_fer(FER_TEST_DIR)

# -------- Cargar RAF-DB --------
def create_dataframe_raf(dir):
    image_paths = []
    labels = []
    for label in os.listdir(dir):
        label_path = os.path.join(dir, label)
        if os.path.isdir(label_path) and label in raf_emotion_map:
            for image_name in os.listdir(label_path):
                image_paths.append(os.path.join(label_path, image_name))
                labels.append(raf_emotion_map[label])
    return image_paths, labels

raf_train = pd.DataFrame()
raf_train['image'], raf_train['label'] = create_dataframe_raf(RAF_TRAIN_DIR)
raf_test = pd.DataFrame()
raf_test['image'], raf_test['label'] = create_dataframe_raf(RAF_TEST_DIR)

# -------- Unifica clases --------
# Encuentra las clases presentes en ambos datasets
fer_classes = set(fer_train['label'].unique())
raf_classes = set(raf_train['label'].unique())
common_classes = list(fer_classes.intersection(raf_classes))

# Filtra ambos datasets a clases comunes
fer_train = fer_train[fer_train['label'].isin(common_classes)]
raf_train = raf_train[raf_train['label'].isin(common_classes)]
fer_test = fer_test[fer_test['label'].isin(common_classes)]
raf_test = raf_test[raf_test['label'].isin(common_classes)]

import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img
import numpy as np

def graficar_muestras(df, titulo, color_mode='grayscale', n=20):
    plt.figure(figsize=(15, 3))
    for i in range(n):
        img_path = df.iloc[i]['image']
        etiqueta = df.iloc[i]['label']
        img = load_img(img_path, color_mode=color_mode, target_size=(48,48))
        img = np.array(img)
        plt.subplot(1, n, i+1)
        plt.imshow(img.squeeze(), cmap='gray' if color_mode=='grayscale' else None)
        plt.title(str(etiqueta))
        plt.axis('off')
    plt.suptitle(titulo)
    plt.show()

# Ejemplo: 10 imágenes del FER2013
graficar_muestras(fer_train, "FER2013 - Ejemplos", color_mode='grayscale', n=10)

# Ejemplo: 10 imágenes del RAF-DB
graficar_muestras(raf_train, "RAF-DB - Ejemplos", color_mode='grayscale', n=10)

# -------- Procesamiento de imágenes --------
def extraer_features(images):
    features = []
    for img_path in tqdm(images):
        img = load_img(img_path, color_mode='grayscale', target_size=(48, 48))
        img = np.array(img)
        features.append(img)
    features = np.array(features)
    features = features.reshape(len(features), 48, 48, 1)
    return features

# Procesar train y test
fer_train_features = extraer_features(fer_train['image'])
raf_train_features = extraer_features(raf_train['image'])
fer_test_features = extraer_features(fer_test['image'])
raf_test_features = extraer_features(raf_test['image'])

# Normaliza
fer_train_features = fer_train_features / 255.0
raf_train_features = raf_train_features / 255.0
fer_test_features = fer_test_features / 255.0
raf_test_features = raf_test_features / 255.0

# -------- Une los datasets --------
x_train = np.concatenate([fer_train_features, raf_train_features], axis=0)
x_test = np.concatenate([fer_test_features, raf_test_features], axis=0)
train_labels = np.concatenate([fer_train['label'].values, raf_train['label'].values], axis=0)
test_labels = np.concatenate([fer_test['label'].values, raf_test['label'].values], axis=0)

# -------- Codifica las etiquetas --------
LE = LabelEncoder()
LE.fit(np.concatenate([train_labels, test_labels], axis=0))
y_train = LE.transform(train_labels)
y_test = LE.transform(test_labels)
num_classes = len(LE.classes_)
y_train = to_categorical(y_train, num_classes=num_classes)
y_test = to_categorical(y_test, num_classes=num_classes)

print(LE.classes_)

# Mostrar 10 imágenes aleatorias del dataset combinado (x_train y train_labels)
import matplotlib.pyplot as plt
import numpy as np

# Si train_labels está codificado como enteros, recupera nombres así:
etiquetas = LE.inverse_transform(np.arange(len(LE.classes_)))

plt.figure(figsize=(15, 3))
idxs = np.random.choice(len(x_train), 10, replace=False)
for i, idx in enumerate(idxs):
    img = x_train[idx].squeeze()  # (48,48)
    # Si tienes train_labels como enteros:
    nombre_etiqueta = LE.inverse_transform([np.argmax(y_train[idx])])[0]
    plt.subplot(1, 10, i+1)
    plt.imshow(img, cmap='gray')
    plt.title(nombre_etiqueta)
    plt.axis('off')
plt.suptitle("Muestras del dataset combinado")
plt.show()

# Análisis de distribución de clases
class_counts = np.sum(y_train, axis=0)
class_names = LE.classes_

plt.figure(figsize=(10, 5))
plt.bar(class_names, class_counts)
plt.title('Distribución de Clases en Training Set')
plt.xlabel('Emoción')
plt.ylabel('Cantidad')
plt.xticks(rotation=45)
plt.show()

# -------- Data Augmentation --------
datagen = ImageDataGenerator(
    rotation_range=7,
    width_shift_range=0.05,
    height_shift_range=0.05,
    shear_range=0.05,
    zoom_range=0.05,
    horizontal_flip=True
)
datagen.fit(fer_train_features)
plt.figure(figsize=(20,8))

for imagen, etiqueta in datagen.flow(x_train, y_train, batch_size=10, shuffle=False):
  for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(imagen[i].reshape(48, 48), cmap="gray")
  break

model = Sequential()
model.add(Input(shape=(48, 48, 1)))
model.add(Conv2D(64, kernel_size=(3, 3), padding='same'))
model.add(BatchNormalization())
model.add(keras.layers.Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))

model.add(Conv2D(128, kernel_size=(3, 3), padding='same'))
model.add(BatchNormalization())
model.add(keras.layers.Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))

model.add(Conv2D(256, kernel_size=(3, 3), padding='same'))
model.add(BatchNormalization())
model.add(keras.layers.Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(128))
model.add(BatchNormalization())
model.add(keras.layers.Activation('relu'))
model.add(Dropout(0.4))
model.add(Dense(num_classes, activation='softmax'))

# Cambia la tasa de aprendizaje inicial


model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

callbacks = [
    EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
]

callbacks.append(ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6))
# -------- Entrenamiento con data augmentation --------
batch_size = 32
epochs = 100

data_gen_entrenamiento = datagen.flow(x_train, y_train, batch_size=32)

history = model.fit(
    data_gen_entrenamiento,
    epochs=epochs,
    validation_data=(x_test, y_test),
    callbacks=callbacks
)

from google.colab import files

import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Validación')
plt.xlabel('Época')
plt.ylabel('Precisión')
plt.legend()
plt.title('Evolución de precisión')
plt.show()

y_pred_probs = model.predict(x_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10,8))
sns.heatmap(cm,
            annot=True,
            fmt="d",
            cmap="Blues")
plt.xlabel('Predicho')
plt.ylabel('Real')
plt.title('Matriz de Confusion')
plt.show()

from google.colab import files

model_json = model.to_json()

with open("emociondetector4.json", "w") as json_file:
    json_file.write(model_json)
model.save("emociondetector4.h5")
print("Modelo guardado")
files.download('/content/emociondetector4.json')
files.download('/content/emociondetector4.h5')

