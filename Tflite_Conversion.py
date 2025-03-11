#Source: ChatGPT
import tensorflow as tf

saved_model_dir = 'D:\Trustworthy final project\saved_model'

# Convert the SavedModel to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)


# Convert and save the TFLite model
tflite_model = converter.convert()


tflite_file_name = 'Pascol_Voc.tflite'
with open(tflite_file_name, 'wb') as f:
    f.write(tflite_model)
