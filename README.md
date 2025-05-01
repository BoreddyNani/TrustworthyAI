# TrustworthyAI
We are developing a android mobile object identification application that will help visually impared by giving an voice output about the detected object.

Dataset: We are using PASCAL VOC dataset
The main idea behind making object detection or object model is Transfer Learning which means using an eficient pre-trained model. Here we used Object Detection API provided by Tensorow (uses SSD mobilenet v1)
The model is trained using SGD with an initial learning rate of 0.001, 0.9 momentum,0.0005 weight decay, and batch size 32.
we exported the pre trained model to the SavedModel format, which is required for conversion to TensorFlow Lite.
Then converted it to .tflite file.
Code to convert the savednodel to .tflite file is given.

Now developing an android application in android studio
Open Android studio,click on file, click on new , click on from version control, copy the git link provided and paste it. 
Gradle dependencies will start installing,Once the graddle dependencies are installed, you can run the project by clicking the green play button.

To test the application download the apk file provided in the git link into your android device and open it and provide the required permissions.

