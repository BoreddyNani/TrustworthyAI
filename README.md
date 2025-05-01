# TrustworthyAI
We are developing a android mobile object identification application that will help visually impaired by giving an voice output about the detected object.

Dataset: We are using PASCAL VOC dataset
The main idea behind making object detection or object model is Transfer Learning which means using an efficient pre-trained model. Here we used Object Detection API provided by Tensorflow (uses SSD mobilenet v1)
The model is trained using SGD with an initial learning rate of 0.001, 0.9 momentum,0.0005 weight decay, and batch size 32.
we exported the pre trained model to the SavedModel format, which is required for conversion to TensorFlow Lite.
Then converted it to .tflite file. 
To attain the pre trained model we used the python files given in python folder
Install the required packages using:
```bash
pip install -r requirements.txt
```
The dataset is organized in the `archive` directory with the following structure:
- `PASCAL_VOC/` - JSON annotations
- `VOCtrainval_06-Nov-2007/` - Training and validation images
- `VOCtest_06-Nov-2007/` - Test images

To train the model from scratch:
```bash
python train.py --batch_size 32 --epochs 100 --learning_rate 0.001
```
To convert the trained model to TensorFlow Lite format for deployment on mobile or edge devices:
```bash
python convert_to_tflite.py --model_path output/ssd_mobilenet_inference.keras --output_path output/lite-model_metadata_2.tflite
```

Now developing an android application in android studio
- Open Android studio
- click on file 
- click on new 
- click on project from version control 
- copy the git link provided and paste it. Gradle dependencies will start installing,
Once the gradle dependencies are installed, you can run the project by clicking the green play button.

To test the application download the apk file provided in the git link into your android device and open it and provide the required permissions.

