import os
# Tensorflow tools
import tensorflow as tf
# For image processsing
from tensorflow.keras.preprocessing.image import (load_img,
                                                  img_to_array,
                                                  ImageDataGenerator)
# Importing VGG16 model
from tensorflow.keras.applications.vgg16 import (preprocess_input,
                                                 decode_predictions,
                                                 VGG16)
# Importing layers
from tensorflow.keras.layers import (Flatten,
                                     Dense,
                                     Dropout,
                                     BatchNormalization)
# Generic model object
from tensorflow.keras.models import Model
# Optimizers
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import SGD
# Tools from scikit-learn
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
# Packages for plotting
import numpy as np
import matplotlib.pyplot as plt
# Packages for working with json files
import pandas as pd
import json
# plotting function from utils 
import sys
sys.path.append(".")
import utils.plotting as pl

def get_json_data():
    # Loading the json metadata. The metadata includes the labels of each class
    test_df = pd.read_json(os.path.join("in", "images", "metadata","test_data.json" ), lines=True)
    train_df = pd.read_json(os.path.join("in", "images", "metadata","train_data.json" ), lines=True)
    val_df = pd.read_json(os.path.join("in", "images", "metadata","val_data.json" ), lines=True)
    
    # Datagenerator train data
    train_datagen = ImageDataGenerator(horizontal_flip=True,
                                        rotation_range=20,
                                        rescale=1/255)
    # Datagenerator test data
    test_datagen = ImageDataGenerator(rescale=1./255.)
    img_dir = os.path.join("in")
    TARGET_size = (224, 224)
    BATCH_size = 32

    # Generating training data
    train_images = train_datagen.flow_from_dataframe(
        dataframe = train_df,
        directory = img_dir,
        x_col ='image_path',
        y_col ='class_label',
        target_size = TARGET_size,
        color_mode ='rgb',
        class_mode ='categorical',
        batch_size = BATCH_size,
        shuffle = True,
        seed = 42,
        subset ='training'
    )
    # Generating test data
    test_images = test_datagen.flow_from_dataframe(
        dataframe = test_df,
        directory = img_dir,
        x_col ='image_path',
        y_col ='class_label',
        target_size = TARGET_size,
        color_mode ='rgb',
        class_mode ='categorical',
        batch_size = BATCH_size,
        shuffle = False
    )
    # Generatig validation data
    val_images = train_datagen.flow_from_dataframe(
        dataframe = val_df,
        directory = img_dir,
        x_col ='image_path',
        y_col ='class_label',
        target_size = TARGET_size,
        color_mode ='rgb',
        class_mode ='categorical',
        batch_size = BATCH_size,
        shuffle = True,
        seed = 42,
    )
    return train_images, test_images, val_images, test_df

# Building the training classifier 
def train_classifier(train_images, test_images, val_images):
    # Loading the VGG16 model without classifier layers
    model = VGG16(include_top=False,
                pooling='avg',
                input_shape=(224, 224, 3)) # Shape set to fit the size of the indo images which was defined above
    # Marking loaded layers as not trainable (freeze all weights)
    for layer in model.layers:
        layer.trainable = False
    # Adding new classifier layers
    flat1 = Flatten()(model.layers[-1].output) # Flatten the images
    bn = BatchNormalization()(flat1) # Batch normalization layer
    class1 = Dense(256,
                activation ='relu')(bn)
    class2 = Dense(128,
                activation = 'relu')(class1)
    output = Dense(15,
                activation ='softmax')(class2)
    # Defining new model
    model = Model(inputs = model.inputs,
                outputs = output)
    # Compiling
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.01,
        decay_steps=10000,
        decay_rate=0.9)
    sgd = SGD(learning_rate=lr_schedule)
    model.compile(optimizer=sgd,
                loss='categorical_crossentropy',
                metrics=['accuracy'])
    # Summarizing the model
    print(model.summary())
    # Fitting model to indo fashion and train model in order to start finetuning 
    history = model.fit(
        train_images,
        # Indicating that the model should train on all the data
        steps_per_epoch = len(train_images),
        validation_data  = val_images,
        validation_steps = len(val_images),
        epochs = 10)
    return history, model
    
def main():
    # Loading and preparing data step
    train_images, test_images, val_images, test_df = get_json_data()
    # Training pretrained CNN
    history, model = train_classifier(train_images, test_images, val_images)
    # Training and validation history plots using helper function from Utils
    pl.plot_history(history, 10)
    output_path = os.path.join("out", "train_and_val_plots.png")
    plt.savefig(output_path, dpi = 100)
    print("Plot is saved!")
    # Predictions
    pred = model.predict(test_images)
    pred = np.argmax(pred,axis=1)
    # Mapping the label
    labels = (train_images.class_indices)
    labels = dict((v,k) for k,v in labels.items())
    pred = [labels[k] for k in pred]
    # Creating classification report
    y_test = list(test_df.class_label)
    report = classification_report(y_test, pred)
    # Saving report in “out” folder
    file_path = os.path.join("out", "classification_report.txt")
    with open(file_path, "w") as f: # “writing” classifier report and saving it
        f.write(report)
    

    
if __name__ == "__main__":
    main()
