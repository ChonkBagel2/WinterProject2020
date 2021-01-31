import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Flatten, Dense
from tensorflow.keras import optimizers
from keras.models import Model
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score


img_size = 64

batches = 32

augment = ImageDataGenerator(rescale = 1. / 255, shear_range = 0.2 , zoom_range = 0.2 , horizontal_flip = True )

training_data = augment.flow_from_directory(r'../input/5vehiclecategories/FinalDataset/FinalDataset' , target_size = (img_size , img_size),
                                            batch_size = batches )

test_data = augment.flow_from_directory(r'../input/5vehiclecategories/FinalTestDataset/FinalTestDataset' , target_size = (img_size , img_size),
                                        batch_size = batches )


vgg = VGG16( include_top = False , input_shape = (img_size , img_size , 3))

for layers in vgg.layers:
    layers.trainable = False
    
x = Flatten()(vgg.output)


x = Dense(120 , activation = 'relu')(x)
x = Dropout(0.2)(x)
x = Dense(120 , activation = 'relu')(x)
last_layer = Dense(5 , activation = 'softmax')(x)


tcnn = Model(inputs = vgg.input , outputs = last_layer)

tcnn.summary()

tcnn.compile(optimizer = 'Adam' , loss = 'categorical_crossentropy' , metrics = ['accuracy'])

tcnn.fit( x = training_data , validation_data = test_data , epochs = 15)

tcnn.save(r'5VehCatSave2.h5')

#for single prediction
Vehicle_dict = {0: 'Bicycle', 1: 'Bus', 2: 'Car', 3: 'Motorbike', 4: 'Truck'}


def predictor(img_path):
    img = image.load_img(img_path , target_size = (64,64))
    img = image.img_to_array(img)
    img = np.expand_dims(img , axis = 0)
    return img

image1 = predictor(r'../input/5vehiclecategories/FinalTestDataset/FinalTestDataset/Bus/13431.jpg')
pred = tcnn.predict(image1)
Vehicle_index = np.argmax(pred)


print(pred)
print(f'Vehicle is {Vehicle_dict[Vehicle_index]}')