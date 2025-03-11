# TrustworthyAI
We are developing a android mobile object identification application that will help visually impared by giving an voice output about the detected object.

--Mid Term

Dataset: We are using PASCAL VOC dataset
Preparing dataset: We prepared the data set by splitting the dataset into training and test set. 
Dataset resource: https://www.kaggle.com/datasets/gopalbhattrai/pascal-voc-2012-dataset?resource=download
The main idea behind making object detection or object classication model is Transfer Learning which means using an ecient pre-trained model. Here we used Object Detection API provided by Tensorow (uses SSD mobilenet v1)
The model is trained using SGD with an initial learning rate of 0.001, 0.9 momentum,0.0005 weight decay, and batch size 32.
Once training is complete, we exported the trained model to the SavedModel format, which is required for conversion to TensorFlow Lite.
Then converted it to .tflite file.
