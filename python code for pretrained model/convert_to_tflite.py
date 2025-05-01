import tensorflow as tf
import numpy as np
import os
import argparse
from models.ssd_mobilenet import SSDMobileNet
from utils.voc_parser import VOC_CLASSES

def convert_to_tflite(model_path, output_path, input_shape=(300, 300, 3), quantize=True):
    """
    Convert Keras model to TensorFlow Lite format
    
    Args:
        model_path: Path to the trained Keras model (.keras or .h5)
        output_path: Path to save the TFLite model
        input_shape: Input shape for the model (height, width, channels)
        quantize: Whether to quantize the model for smaller size and faster inference
    """
    print(f"Loading model from {model_path}...")
    
    # Load the trained model
    try:
        # First try loading as a saved model with custom objects
        model = tf.keras.models.load_model(
            model_path,
            custom_objects={
                'compute_loss': SSDMobileNet(input_shape=input_shape).ssd_loss()
            },
            compile=False
        )
        print("Model loaded successfully as a saved model with custom objects")
    except:
        try:
            # Try loading as a regular saved model
            model = tf.keras.models.load_model(model_path, compile=False)
            print("Model loaded successfully as a regular saved model")
        except:
            # Try recreating the inference model and loading weights
            print("Recreating inference model and loading weights...")
            ssd_model = SSDMobileNet(input_shape=input_shape)
            model = ssd_model.build_inference_model()
            model.load_weights(model_path)
            print("Model loaded successfully by recreating inference model")
    
    # Create a concrete function from the model
    # This is necessary to ensure the model works with TensorFlow Lite
    print("Creating concrete function...")
    
    # Create a wrapper function that returns a dictionary of outputs
    class DetectionModel(tf.Module):
        def __init__(self, model):
            super(DetectionModel, self).__init__()
            self.model = model
        
        @tf.function(input_signature=[tf.TensorSpec(shape=[1, input_shape[0], input_shape[1], 3], dtype=tf.float32)])
        def __call__(self, images):
            # Preprocess the input (normalize to [-1, 1])
            preprocessed = tf.cast(images, dtype=tf.float32) / 127.5 - 1.0
            
            # Run inference
            boxes, class_probs = self.model(preprocessed, training=False)
            
            # Process outputs for easier use in Android
            # Get class with highest probability for each detection
            classes = tf.argmax(class_probs, axis=-1)
            scores = tf.reduce_max(class_probs, axis=-1)
            
            # Get the number of detections (fixed for SSD models)
            num_detections = tf.shape(boxes)[1]
            
            # Return as a dictionary for easier access in Android
            return {
                'detection_boxes': boxes,
                'detection_classes': tf.cast(classes, tf.float32),
                'detection_scores': scores,
                'num_detections': tf.cast(num_detections, tf.float32)
            }
    
    detection_module = DetectionModel(model)
    concrete_function = detection_module.__call__.get_concrete_function()
    
    # Create a TensorFlow Lite converter
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_function])
    
    # Set optimization options
    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        # For full integer quantization, you would need representative dataset
        # This is just weight quantization
    
    # Set options to ensure compatibility with TensorFlow Lite
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,  # Use TFLite built-in ops
        tf.lite.OpsSet.SELECT_TF_OPS     # Allow some TF ops if needed
    ]
    
    # Convert the model
    print("Converting model to TFLite format...")
    tflite_model = converter.convert()
    
    # Save the model
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"Model saved to {output_path}")
    
    # Save metadata file with class labels
    metadata_path = output_path.replace('.tflite', '_labels.txt')
    with open(metadata_path, 'w') as f:
        for class_name in VOC_CLASSES:
            f.write(f"{class_name}\n")
    
    print(f"Class labels saved to {metadata_path}")
    
    # Print model size
    model_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"TFLite model size: {model_size_mb:.2f} MB")
    
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Keras model to TFLite")
    parser.add_argument("--model-path", required=True, help="Path to the Keras model file (.keras or .h5)")
    parser.add_argument("--output-path", default="output/ssd_mobilenet.tflite", help="Path to save the TFLite model")
    parser.add_argument("--input-height", type=int, default=300, help="Input image height")
    parser.add_argument("--input-width", type=int, default=300, help="Input image width")
    parser.add_argument("--no-quantize", action="store_true", help="Disable quantization")
    
    args = parser.parse_args()
    
    input_shape = (args.input_height, args.input_width, 3)
    convert_to_tflite(args.model_path, args.output_path, input_shape, not args.no_quantize)