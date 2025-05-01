import os
import cv2
import argparse
import numpy as np
import tensorflow as tf
from utils.voc_parser import VOC_CLASSES

def load_tflite_model(model_path):
    """Load a TensorFlow Lite model."""
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def get_input_details(interpreter):
    """Get input details from the TFLite interpreter."""
    input_details = interpreter.get_input_details()
    input_shape = input_details[0]['shape']
    input_dtype = input_details[0]['dtype']
    return input_details, input_shape, input_dtype

def get_output_details(interpreter):
    """Get output details from the TFLite interpreter."""
    output_details = interpreter.get_output_details()
    return output_details

def preprocess_image(image_path, input_shape):
    """Preprocess an image for TFLite SSD MobileNet model inference."""
    # Load image with OpenCV (in BGR format)
    original_image = cv2.imread(image_path)
    if original_image is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Convert to RGB for model input
    rgb_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    
    # Resize for model input
    height, width = input_shape[1:3]
    resized_image = cv2.resize(rgb_image, (width, height))
    
    # Normalize to [-1, 1] as in convert_to_tflite.py
    input_img = resized_image.astype(np.float32) / 127.5 - 1.0
    input_img = np.expand_dims(input_img, axis=0)  # Add batch dimension
    
    return original_image, input_img

def decode_predictions(boxes, scores, classes, confidence_threshold=0.3, nms_threshold=0.4):
    """Apply post-processing to detections."""
    # Print raw values to help with debugging
    print(f"Raw prediction stats:")
    print(f"  Boxes shape: {boxes.shape}")
    print(f"  Scores range: {np.min(scores):.4f} to {np.max(scores):.4f}")
    print(f"  Classes distribution: {np.bincount(classes.astype(np.int32))}")
    
    # Filter by confidence threshold
    mask = scores > confidence_threshold
    boxes = boxes[mask]
    scores = scores[mask]
    classes = classes[mask]
    
    print(f"After confidence threshold {confidence_threshold}:")
    print(f"  Kept {len(boxes)} out of {len(mask)} detections")
    
    if len(boxes) == 0:
        return np.array([]), np.array([]), np.array([])
    
    # Non-max suppression (simplified version)
    selected_indices = tf.image.non_max_suppression(
        boxes, scores, max_output_size=100, iou_threshold=nms_threshold
    )
    
    selected_boxes = tf.gather(boxes, selected_indices).numpy()
    selected_scores = tf.gather(scores, selected_indices).numpy()
    selected_classes = tf.gather(classes, selected_indices).numpy()
    
    print(f"After NMS:")
    print(f"  Final detections: {len(selected_boxes)}")
    if len(selected_boxes) > 0:
        print(f"  Class distribution: {np.bincount(selected_classes.astype(np.int32))}")
    
    return selected_boxes, selected_scores, selected_classes

def draw_detections(image, boxes, scores, classes, class_names):
    """Draw detection boxes on the image using OpenCV."""
    # Make a copy of the image to avoid modifying the original
    image_with_boxes = image.copy()
    
    # Define colors for different classes (in BGR format for OpenCV)
    num_classes = len(class_names)
    colors = []
    for i in range(num_classes):
        # Generate distinct colors
        hue = int(255 * i / num_classes)
        # HSV color with full saturation and value
        hsv_color = np.array([[[hue, 255, 255]]], dtype=np.uint8)
        # Convert to BGR
        bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
        # Convert to int tuple
        color = (int(bgr_color[0]), int(bgr_color[1]), int(bgr_color[2]))
        colors.append(color)
    
    height, width = image_with_boxes.shape[:2]
    
    for i, (box, score, cls) in enumerate(zip(boxes, scores, classes)):
        # Get class name and color
        class_idx = int(cls)
        class_name = class_names[class_idx]
        color = colors[class_idx % len(colors)]
        
        # Convert normalized coordinates to pixel coordinates
        ymin, xmin, ymax, xmax = box  # Note: TFLite model outputs [ymin, xmin, ymax, xmax]
        xmin = max(0, int(xmin * width))
        xmax = min(width, int(xmax * width))
        ymin = max(0, int(ymin * height))
        ymax = min(height, int(ymax * height))
        
        # Skip invalid boxes
        if xmin >= xmax or ymin >= ymax:
            continue
            
        # Draw rectangle with class color
        cv2.rectangle(image_with_boxes, (xmin, ymin), (xmax, ymax), color, 3)
        
        # Prepare text
        label = f"{class_name}: {score:.2f}"
        
        # Get text size
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        
        # Draw label background
        cv2.rectangle(
            image_with_boxes, 
            (xmin, ymin - text_height - 10), 
            (xmin + text_width, ymin), 
            color, 
            -1
        )
        
        # Draw text
        cv2.putText(
            image_with_boxes, 
            label, 
            (xmin, ymin - 5), 
            font, 
            font_scale, 
            (255, 255, 255), 
            thickness
        )
    
    return image_with_boxes

def detect_objects_tflite(interpreter, image_path, class_names, confidence_threshold=0.3):
    """Run object detection on a single image using TFLite model."""
    # Get model details
    input_details, input_shape, input_dtype = get_input_details(interpreter)
    output_details = get_output_details(interpreter)
    
    # Print output tensor details for debugging
    print("Available output tensors:")
    for i, output in enumerate(output_details):
        print(f"  {i}: {output['name']} (shape: {output['shape']})")
    
    # Load and preprocess image
    original_image, input_img = preprocess_image(image_path, input_shape)
    
    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], input_img.astype(input_dtype))
    
    # Run inference
    print("Running TFLite inference...")
    interpreter.invoke()
    
    # Get all output tensors by index
    outputs = []
    for i in range(len(output_details)):
        tensor = interpreter.get_tensor(output_details[i]['index'])
        outputs.append(tensor)
        print(f"Output {i}: shape={tensor.shape}, min={np.min(tensor):.4f}, max={np.max(tensor):.4f}")
    
    # Try to identify which tensor is which based on shape and content
    boxes_tensor = None
    scores_tensor = None
    classes_tensor = None
    
    for tensor in outputs:
        # Boxes typically have shape [1, num_boxes, 4]
        if len(tensor.shape) == 3 and tensor.shape[2] == 4:
            boxes_tensor = tensor
        # Scores typically have shape [1, num_boxes] and values between 0 and 1
        elif len(tensor.shape) == 2 and 0 <= np.max(tensor) <= 1:
            scores_tensor = tensor
        # Classes typically have shape [1, num_boxes] and integer-like values
        elif len(tensor.shape) == 2 and np.allclose(tensor, tensor.astype(int)):
            classes_tensor = tensor
    
    # If we couldn't identify tensors by shape/values, fall back to assuming order
    if boxes_tensor is None and len(outputs) > 0:
        print("Warning: Could not identify boxes tensor, using first output")
        boxes_tensor = outputs[0]
    if scores_tensor is None and len(outputs) > 1:
        print("Warning: Could not identify scores tensor, using second output")
        scores_tensor = outputs[1]
    if classes_tensor is None and len(outputs) > 2:
        print("Warning: Could not identify classes tensor, using third output")
        classes_tensor = outputs[2]
    
    # Make sure we have at least one output tensor
    if boxes_tensor is None or len(outputs) == 0:
        raise ValueError("Could not identify any output tensors")
    
    # Remove batch dimension if present
    boxes = boxes_tensor[0] if boxes_tensor.shape[0] == 1 else boxes_tensor
    
    # Handle scores and classes if available
    if scores_tensor is not None:
        scores = scores_tensor[0] if scores_tensor.shape[0] == 1 else scores_tensor
    else:
        # Generate dummy scores if not available
        scores = np.ones(boxes.shape[0])
        
    if classes_tensor is not None:
        classes = classes_tensor[0] if classes_tensor.shape[0] == 1 else classes_tensor
    else:
        # Generate dummy classes if not available
        classes = np.zeros(boxes.shape[0])
    
    # Post-process detections
    selected_boxes, selected_scores, selected_classes = decode_predictions(
        boxes, scores, classes, confidence_threshold
    )
    
    # Draw detections on the image
    result_image = draw_detections(
        original_image, 
        selected_boxes, 
        selected_scores, 
        selected_classes, 
        class_names
    )
    
    return result_image, selected_boxes, selected_scores, selected_classes

def load_class_names(labels_path):
    """Load class names from a text file."""
    with open(labels_path, 'r') as f:
        return [line.strip() for line in f.readlines()]

def main():
    parser = argparse.ArgumentParser(description='Object detection with SSD MobileNet TFLite')
    parser.add_argument('--model-path', type=str, default='output/ssd_mobilenet.tflite',
                       help='Path to the TFLite model')
    parser.add_argument('--labels-path', type=str, default='output/ssd_mobilenet_labels.txt',
                       help='Path to the labels file')
    parser.add_argument('--image-path', type=str, required=True,
                       help='Path to the input image for detection')
    parser.add_argument('--output-path', type=str, default=None,
                       help='Path to save the output image (default: detection_output.jpg)')
    parser.add_argument('--confidence-threshold', type=float, default=0.3,
                       help='Confidence threshold for filtering detections')
    parser.add_argument('--display', action='store_true',
                       help='Display the result image in a window (requires GUI)')
    
    args = parser.parse_args()
    
    # Set default output path if not provided
    if args.output_path is None:
        base_name = os.path.basename(args.image_path)
        args.output_path = f"detection_tflite_{base_name}"
    
    # Load model
    print(f"Loading TFLite model from {args.model_path}...")
    interpreter = load_tflite_model(args.model_path)
    
    # Load class names
    print(f"Loading class names from {args.labels_path}...")
    class_names = load_class_names(args.labels_path)
    
    # Run detection
    print(f"Running detection on {args.image_path}...")
    result_image, boxes, scores, classes = detect_objects_tflite(
        interpreter, 
        args.image_path, 
        class_names, 
        confidence_threshold=args.confidence_threshold
    )
    
    # Save the result
    print(f"Saving result to {args.output_path}...")
    cv2.imwrite(args.output_path, result_image)
    
    # Display the result if requested
    if args.display:
        cv2.imshow('Detection Result', result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    # Print detection results
    print("\nDetection Results:")
    if len(boxes) == 0:
        print("No objects detected.")
    else:
        for i, (box, score, cls) in enumerate(zip(boxes, scores, classes)):
            class_idx = int(cls)
            class_name = class_names[class_idx]
            print(f"  {i+1}. {class_name}: {score:.4f}, box: {box}")

if __name__ == "__main__":
    main()