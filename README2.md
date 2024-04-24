# Tomato Classifier CNN

Este proyecto tiene como objetivo detectar con la mayor precisión posible las lesiones en las hojas de plantas de tomates y distinguir entre 7 categorías diferentes.

## Dataset

El conjunto de datos original incluye 737 imágenes de hojas de tomate con lesiones. Las lesiones están anotadas en formato YOLO v5 PyTorch. Las imágenes han sido preprocesadas de la siguiente manera:

- Orientación automática de los datos de píxeles (con eliminación de la orientación EXIF).
- Redimensionamiento a 640x640 (estiramiento).



## Fuentes de Datos

El conjunto de datos proviene de:

- [Tomato Leaf Diseases Detection - Kaggle](https://www.kaggle.com/datasets/farukalam/tomato-leaf-diseases-detection-computer-vision)
- [Tomato Leaf Diseases Detect - Roboflow](https://universe.roboflow.com/sylhet-agricultural-university/tomato-leaf-diseases-detect)

Proporcionado por un usuario de Roboflow.

## Categorías de Lesiones
1. Bacterial Spot (Manchas Bacterianas)
Las manchas bacterianas en las hojas de tomate, también conocidas como manchas bacterianas del tomate, son causadas por la bacteria Xanthomonas campestris pv. vesicatoria. Esta enfermedad es común en áreas con clima cálido y húmedo, y puede causar daños significativos en los cultivos de tomate. Las lesiones aparecen como pequeñas manchas de agua en las hojas, que eventualmente se convierten en manchas necróticas con un halo amarillo. Si la enfermedad es severa, puede provocar la defoliación de la planta y la pérdida de rendimiento.

2. Early Blight (Mildiú Temprano)
El mildiú temprano es causado por el hongo Alternaria solani y puede infectar las hojas de tomate en cualquier momento durante el ciclo de crecimiento de la planta. Esta enfermedad se caracteriza por lesiones irregulares cerca del suelo, que desarrollan parches amarillos que se oscurecen en anillos concéntricos negros y pueden tener una región clorótica alrededor de la lesión.

3. Healthy (Sano)
Las hojas sanas tienen vigor, un color uniforme (a menos que sean variegadas), crecimiento abierto y una apariencia erguida.

4. Late Blight (Mildiú Tardío)
El mildiú tardío, causado por el hongo Phytophthora infestans, es una de las enfermedades más devastadoras para las hojas de tomate a nivel mundial, causando grandes pérdidas económicas anuales. Se detecta típicamente en las hojas recién desarrolladas en la parte superior de la planta, con lesiones irregulares y encharcadas como los primeros signos. A medida que empeora, las lesiones se vuelven más grandes y las hojas afectadas se vuelven marrones, marchitas y mueren.

5. Leaf Mold (Moho en las Hojas)
El moho en las hojas del tomate es causado por el hongo Passalora fulva. Se caracteriza por pequeñas manchas redondas, verde-amarillentas y borrosas en la parte superior de las hojas. El hongo se asienta en las hojas y penetra en los estomas de la planta, que se utilizan para el intercambio de gases.

6. Target Spot (Mancha Objetivo)
La mancha objetivo en las hojas de tomate es causada por el hongo Corynespora cassiicola. Se desarrolla en regiones con clima cálido durante todo el año y se manifiesta inicialmente como pequeñas manchas llenas de agua en las hojas. Las manchas se convierten en pequeñas lesiones necróticas con centros marrones claros y márgenes oscuros. Estas infecciones pueden reducir la producción indirectamente al disminuir el área fotosintética y directamente al hacer que el fruto sea menos comercializable debido a las manchas en el fruto.

7. Black Spot (Manchas Negras)
Las manchas negras en las hojas de tomate pueden ser causadas por varias enfermedades fúngicas, como Alternaria alternata y Alternaria solani. Estas enfermedades suelen desarrollarse en condiciones de alta humedad y pueden causar lesiones necróticas en las hojas de tomate. Las manchas negras pueden variar en tamaño y forma, y si no se controlan adecuadamente, pueden provocar la defoliación de la planta y la reducción del rendimiento del cultivo.

## Preprocesamiento de Imágenes

Las imágenes han sido preprocesadas utilizando OpenCV para eliminar reflejos y sombras, lo que ayuda a mejorar la calidad de los datos y la precisión del modelo. Además, las imágenes han sido redimensionadas a 128x128 píxeles para mejorar la eficiencia del entrenamiento. Además, los valores de píxeles han sido normalizados para que estén en el rango de 0 a 1 mediante la división por 255.0.

## Modelo

Se utiliza una red neuronal convolucional (CNN) para este proyecto, ya que estas capas permiten extraer los features más importantes de las imágenes. La arquitectura y los detalles específicos del modelo se agregarán en futuras actualizaciones.

## División de Datos

Los datos ya vienen divididos en conjuntos de entrenamiento, validación y prueba.

## Entrenamiento del Modelo

Se utiliza el optimizador Adam y la función de pérdida de entropía cruzada categórica (categorical cross entropy) para el entrenamiento del modelo. La métrica de evaluación principal es la matriz de confusión.

