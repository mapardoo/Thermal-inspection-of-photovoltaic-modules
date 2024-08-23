import os
import numpy as np
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical, Sequence
import cv2
from utils.carga_metadatos import cargar_metadatos
import pickle

# Función para obtener etiquetas one-hot
def get_one_hot_labels(keys, metadata, label_encoder, num_classes):
    # Obtiene las etiquetas de clase de anomalía a partir de los metadatos usando las claves proporcionadas
    labels = [metadata[key]['anomaly_class'] for key in keys]
    # Codifica las etiquetas usando el codificador de etiquetas proporcionado
    labels_encoded = label_encoder.transform(labels)
    # Convierte las etiquetas codificadas a formato one-hot
    return to_categorical(labels_encoded, num_classes=num_classes)

# Clase para generar datos en lotes de manera secuencial
class DataGenerator(Sequence):
    def __init__(self, keys, metadata, image_folder, label_encoder, num_classes=12, batch_size=32, dim=(40, 24), n_channels=1, shuffle=True):
        # Inicializa los parámetros del generador de datos
        self.keys = keys
        self.metadata = metadata
        self.image_folder = image_folder
        self.label_encoder = label_encoder
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()  # Inicializa los índices y los baraja si es necesario

    def __len__(self):
        # Devuelve el número de lotes por época
        return int(np.ceil(len(self.keys) / self.batch_size))

    def __getitem__(self, index):
        # Genera un lote de datos en el índice dado
        batch_keys = self.keys[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = self.__data_generation(batch_keys)
        return X, y

    def on_epoch_end(self):
        # Actualiza los índices después de cada época y los baraja si es necesario
        self.indexes = np.arange(len(self.keys))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, batch_keys):
        # Genera datos y etiquetas para el lote proporcionado
        X = np.zeros((len(batch_keys), *self.dim, self.n_channels), dtype=np.float32)
        y = np.zeros((len(batch_keys), self.num_classes), dtype=np.float32)

        for i, key in enumerate(batch_keys):
            # Construye la ruta completa de la imagen
            image_path = os.path.join(self.image_folder, self.metadata[key]['image_filepath'])
            # Carga la imagen en escala de grises
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Asegurarse de cargar la imagen en escala de grises
            img = img.astype(np.float32) / 255.0

            img = np.expand_dims(img, axis=-1)  # Añade una dimensión al final para los canales

            # Asigna la imagen y la etiqueta one-hot al lote
            X[i] = img
            y[i] = get_one_hot_labels([key], self.metadata, self.label_encoder, self.num_classes)[0]

        return X, y

# Función para balancear las clases
def balancear_clases(metadata, label_encoder):
    all_labels = [metadata[key]['anomaly_class'] for key in metadata.keys()]
    label_counts = Counter(all_labels)
    most_common_classes = label_counts.most_common()

    max_class = most_common_classes[0][0]
    second_max_count = most_common_classes[1][1]

    balanced_keys = []
    class_counts = {key: 0 for key in label_counts.keys()}

    for key in metadata.keys():
        anomaly_class = metadata[key]['anomaly_class']
        if anomaly_class == max_class and class_counts[max_class] <= second_max_count:
            balanced_keys.append(key)
            class_counts[max_class] += 1
        elif anomaly_class != max_class:
            balanced_keys.append(key)
            class_counts[anomaly_class] += 1

    return balanced_keys, label_encoder, len(label_encoder.classes_)

# Función para obtener generadores de datos
def obtener_generadores(path_file, image_folder, batch_size=32, test_size=0.2, random_state=42):
    metadata = cargar_metadatos(path_file)

    label_encoder = LabelEncoder()
    all_labels = [metadata[key]['anomaly_class'] for key in metadata.keys()]
    label_encoder.fit(all_labels)

    balanced_keys, _, num_classes = balancear_clases(metadata, label_encoder)

    train_keys, test_keys = train_test_split(balanced_keys, test_size=test_size, random_state=random_state)

    train_generator = DataGenerator(train_keys, metadata, image_folder, label_encoder, num_classes, batch_size=batch_size, shuffle=True)
    test_generator = DataGenerator(test_keys, metadata, image_folder, label_encoder, num_classes, batch_size=batch_size, shuffle=False)

    return train_generator, test_generator, label_encoder, num_classes

# ###################### Just for visualizing #########################
# # Ruta del archivo JSON y la carpeta de imágenes
# path_file = 'C:/Users/USUARIO/Downloads/data_solar/InfraredSolarModules/module_metadata.json'
# image_folder = 'C:/Users/USUARIO/Downloads/data_solar/InfraredSolarModules'

# # Obtener generadores de datos
# train_generator, test_generator = obtener_generadores(path_file, image_folder)

# # Mostrar el número de muestras por cada clase en el conjunto de entrenamiento
# class_counts_train = Counter([train_generator.metadata[key]['anomaly_class'] for key in train_generator.keys])
# print("Número de muestras por clase en el conjunto de entrenamiento:")
# for class_name, count in class_counts_train.items():
#     print(f"Clase {class_name}: {count} muestras")

# # Visualizar una imagen de cada clase
# fig, axes = plt.subplots(nrows=1, ncols=len(class_counts_train), figsize=(15, 5))

# for i, (class_name, _) in enumerate(class_counts_train.items()):
#     # Encontrar la primera imagen de la clase actual
#     for key in train_generator.keys:
#         if train_generator.metadata[key]['anomaly_class'] == class_name:
#             image_path = os.path.join(image_folder, train_generator.metadata[key]['image_filepath'])
#             img = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)  # Cargar en escala de grises
            
#             if img is not None:
#                 img_dimensions = img.shape
#                 print(f"Clase {class_name} - Dimensiones: {img_dimensions}")
                
                    
#                 axes[i].imshow(img.squeeze(), cmap='gray')
#                 axes[i].text(0.5, 1.05, f'{class_name}\n({key})', horizontalalignment='center', verticalalignment='center', transform=axes[i].transAxes, fontsize=10)
#                 axes[i].axis('off')
#                 break

# plt.tight_layout()
# plt.show()
