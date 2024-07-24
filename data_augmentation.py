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
