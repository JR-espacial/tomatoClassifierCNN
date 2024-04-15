# tomatoClassifierCNN

En este proyecto intentaré detectar con la mayor precisión posible las lesiones en las hojas de plantas de tomates y distinguir entre 7 categorías diferentes.

En el archivo load_data, me encargué primeramente de leer los datos y cargar tanto las imágenes como las etiquetas en el formato necesario para entrenar mi modelo.

También normalicé los datos y les di la forma que espera el modelo.

Despues de algunos ejercicios con el modelo me dicuenta de que no aprendia mucho y descubri algunas cosas que mejorar en el modelo:

actualemente estoy trabajando para destacat aun mas las carateristicas mas relevantes eliminar ruido con normalizacion de los colores, buscar eliminar reflejos y detectar contronos de las manchas