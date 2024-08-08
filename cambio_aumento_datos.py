# cambio dentro de clase para hacer el aumento de datos directo
class DataGenerator(Sequence):
    def __init__(self, keys, metadata, image_folder, label_encoder, num_classes=12, batch_size=16, dim=(40, 24), n_channels=1, shuffle=True, augment=False):
        self.keys = keys
        self.metadata = metadata
        self.image_folder = image_folder
        self.label_encoder = label_encoder
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.augment = augment
        self.on_epoch_end()

        if self.augment:
            self.datagen = ImageDataGenerator(
                rotation_range=10,
                horizontal_flip=True,
                fill_mode='nearest'
            )

# cambio para incluir solamente el aumento de datos dentro del generador de entrenamiento
def obtener_generadores(path_file, image_folder, class_mapping, batch_size=32, test_size=0.2, random_state=42, augment=False):
    metadata = cargar_metadatos(path_file, class_mapping)

    label_encoder = LabelEncoder()
    all_labels = [metadata[key]['anomaly_class'] for key in metadata.keys()]
    label_encoder.fit(all_labels)

    balanced_keys, _, num_classes = balancear_clases(metadata, label_encoder)

    train_keys, test_keys = train_test_split(balanced_keys, test_size=test_size, random_state=random_state)

    train_generator = DataGenerator(train_keys, metadata, image_folder, label_encoder, num_classes, batch_size=batch_size, shuffle=True, augment=augment) # se llama aca para hacer el aumento con solo rotacion y flip
    test_generator = DataGenerator(test_keys, metadata, image_folder, label_encoder, num_classes, batch_size=batch_size, shuffle=False)

    return train_generator, test_generator
