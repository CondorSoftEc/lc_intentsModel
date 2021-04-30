# Modelo de clasificación de intenciones
# Acera de este proyecto

En este proyecto, presentamos el modelo de clasificación de intenciones. El cual empieza explicando la estructura de datos que recibe, la transformación de los datos para adecuarlos a nuestro modelo de entrenamiento, como se genera el "validation set", "training set" y "testing set". Para finalmente explicar el modelo y como acceder a este. 

## Instalación

Para usar el siguiente proyecto, una vez clonado instalar las dependencias:

```bash
pip install -r requirements.txt
```
Para subir los modelos al firebase convertirlos a .json con el siguiente comando en el terminal:

```bash
tensorflowjs_converter --input_format=keras /tmp/model.h5 /tmp/tfjs_model
```

## Ejemplo de datos a recibir

Los datos que recibe el modelo, son una tabla con las siguientes columnas:
Intensión| Texto
--- | --- 
ObtenerInformacion | Me ayudas con información acerca del curso?


La primera columna hace referencia a la intensión, mientras que en la columna de texto se provee el texto, o pregunta a la cual pertenece dicha intensión.

# Pipeline del modelo
## Pre-procesamiento de los datos

Es necesario realizar un pre-procesamiento de los datos y de esta manera estandarizarlos. El pre-procesamiento de datos que se encuentra en este modelo es el siguiente.
1. Eliminar símbolos de puntuación y caracteres especiales.
2. Estandarizar toda la letra a minúscula
3. Eliminar todo tipo de acentos (aunque muchas palabras cambien su significado con el uso de tildes en el español, el uso que se le quiere dar al modelo para ser usado dentro de un chabota en el cual muchos usuarios cometen este tipo de faltas ortográficas obliga a tomar esto en consideración) 

## Procesamiento de los datos
Los datos a ser procesados siguen la siguiente ruta:
1. De todos los datos, se dividen en un conjunto de prueba (30%) y uno de entrenamiento (70%)
2. Los datos dentro del conjunto de entrenamiento serán tokenizados con un numero máximo de palabras a tokenizar de 4000
3. Luego se hará que todas las oraciones tengan la misma longitud, para esto se usurará padding. El tamaño de la oración corresponderá a la media del tamaño del percentil 98 de los datos de entrenamiento. 
4. Finalmente se usara el método One-Hot Encoding para clasificación. 

# Modelo de aprendizaje
El modelo de aprendizaje es el siguiente:
```python
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Input, Dropout, LSTM, Activation, Bidirectional,Embedding
model = Sequential()

model.add(Embedding(num_words, 100, trainable=False,input_length=train_sequences.shape[1]))
model.add(Bidirectional(LSTM(256, return_sequences=True, recurrent_dropout=0.1, dropout=0.1), 'concat'))
model.add(Dropout(0.3))
model.add(LSTM(256, return_sequences=False, recurrent_dropout=0.1, dropout=0.1))
model.add(Dropout(0.3))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(classes.shape[0], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
```
# API

Una ves entrenado el modelo se podrá pedir datos de la siguiente manera.  
Dirección base del api:
```bash
https://api.chatbot.com
```
La url debe proveer la versión actual del api (versión actual: v0)

```bash
https://api.chatbot.com/v0/

```
## POST /intent
Retorna la atención del mensaje dado
### Parametros
Nombre| Tipo | Descripción
--- | --- | --- 
msg| string | El mensaje para darle una intensión.
allInfo | boolean (default = false)| Si es verdadero retornará la lista de las posibles intenciones con su probabilidad.

## License
[MIT](https://choosealicense.com/licenses/mit/)
