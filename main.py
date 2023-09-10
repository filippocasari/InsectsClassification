from tensorflow import keras
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import random
from sklearn.model_selection import train_test_split
from pathlib import Path
main_path = 'insects/'

path_butterfly = main_path+'Butterfly/'
path_drangonfly = main_path+'Dragonfly/'
path_grasshopper = main_path+'Grasshopper/'
path_ladybug = main_path+'Ladybug/'
path_mosquito = main_path+'Mosquito/'




def load_data(path):
    data = []
    pathlist = Path(path).rglob('*.jpg')
    #num_img = 0
    for i in pathlist:
        
        path_in_str = str(i)
        
        img = Image.open(path_in_str)
        img_resized = img.resize((224, 224), resample=Image.BILINEAR)
        
        img_resized = np.array(img_resized)
        #img_resized = img_resized/255.0
        #print(img_resized[0, :])

        print("image : ",path_in_str)
        data.append(np.array(img_resized))
            
            
    return data

    
data_butterfly = load_data(path_butterfly)
plt.imshow(data_butterfly[random.randint(0, len(data_butterfly))])

plt.show()

data_drangonfly = load_data(path_drangonfly)
data_grasshopper = load_data(path_grasshopper)
data_ladybug = load_data(path_ladybug)
data_mosquito = load_data(path_mosquito)

img_height = 224
img_width = 224

model = models.Sequential([
  layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(5)
])
model.summary()

y_data  = np.concatenate((np.zeros(len(data_butterfly)), np.ones(len(data_drangonfly)), np.ones(len(data_grasshopper))*2, np.ones(len(data_ladybug))*3, np.ones(len(data_mosquito))*4))
x_data = np.concatenate((data_butterfly, data_drangonfly, data_grasshopper, data_ladybug, data_mosquito))
X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.30, random_state=42)



model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=10, 
                    validation_data=(X_test, y_test))


plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(X_test,  y_test, verbose=2)