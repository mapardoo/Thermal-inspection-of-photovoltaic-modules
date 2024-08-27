# Original classes to new groups
## Note: for this mapping is required to adjust the final model with 8 possible groups
class_mapping = {
    'Vegetation': 'Shadowing',
    'Shadowing': 'Shadowing',
    'No-Anomaly': 'No-Anomaly',
    'Offline-Module': 'Offline-Module',
    'Hot-Spot-Multi': 'Hot-Spot',
    'Hot-Spot': 'Hot-Spot',
    'Diode-Multi': 'Diode',
    'Diode': 'Diode',
    'Soiling': 'Soiling',
    'Cell-Multi': 'Cell',
    'Cell': 'Cell',
    'Cracking': 'Cracking'
}

# Load metadata from JSON. Remapping classes
def cargar_metadatos(path_file, class_mapping):
    with open(path_file, 'r') as file:
        metadata = json.load(file)
    for key in metadata.keys():
        original_class = metadata[key]['anomaly_class']
        if original_class in class_mapping:
            metadata[key]['anomaly_class'] = class_mapping[original_class]
    return metadata
