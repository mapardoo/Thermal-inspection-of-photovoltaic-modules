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
