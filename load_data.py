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
