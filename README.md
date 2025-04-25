# TrustworthyAI
We are developing a android mobile object identification application that will help visually impared by giving an voice output about the detected object.

--Work till mid term

Dataset: We are using PASCAL VOC dataset
Preparing dataset: We prepared the data set by splitting the dataset into training and test set. 
Dataset resource: https://www.kaggle.com/datasets/gopalbhattrai/pascal-voc-2012-dataset?resource=download
The main idea behind making object detection or object model is Transfer Learning which means using an eficient pre-trained model. Here we used Object Detection API provided by Tensorow (uses SSD mobilenet v1)
The model is trained using SGD with an initial learning rate of 0.001, 0.9 momentum,0.0005 weight decay, and batch size 32.
Once training is complete, we exported the trained model to the SavedModel format, which is required for conversion to TensorFlow Lite.
Then converted it to .tflite file.
Code to convert the savednodel to .tflite file is given.

Now developing and android application in android studio
First activity of application is written and tested.
Open android studio and start a new project
import the app files to it
choose java as programming language
Open tools and AVD Manager
Create a new virtual device and start the emulator
Run the code MainActivity1.
The screen should splash.

-- Code work expected to be finished till final term

Complete MainActivity2 where objectdetection is done and audio output is given.
Refine the project.

