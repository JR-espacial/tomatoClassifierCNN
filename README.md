# tomatoClassifierCNN

En este proyecto intentaré detectar con la mayor precisión posible las lesiones en las hojas de plantas de tomates y distinguir entre 7 categorías diferentes.

En el archivo load_data, me encargué primeramente de leer los datos y cargar tanto las imágenes como las etiquetas en el formato necesario para entrenar mi modelo.

También normalicé los datos y les di la forma que espera el modelo.

Dataset:
# Tomato Leaf DIseases Detect > 2024-03-04 3:55pm
https://www.kaggle.com/datasets/farukalam/tomato-leaf-diseases-detection-computer-vision

https://universe.roboflow.com/sylhet-agricultural-university/tomato-leaf-diseases-detect

Provided by a Roboflow user
License: Public Domain


Paper de refercia para mejoras:
  https://www.sciencedirect.com/science/article/pii/S2590005623000383

  - Increase epoch num
  - Add more layers
  - Network architecture that starts with a larger number of filters and gradually reduces the number of filters in deeper layers
  - Drop out 25%