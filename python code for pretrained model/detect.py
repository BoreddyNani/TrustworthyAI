import os
import cv2
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from utils.voc_parser import VOC_CLASSES
from models.ssd_mobilenet import SSDMobileNet

def decode_predictions(boxes, scores, classes, confidence_threshold=0.3, nms_threshold=0.4):
    """Apply post-processing to detections."""
    # Print raw values to help with debugging
    print(f"Raw prediction stats:")
    print(f"  Boxes shape: {boxes.shape}")
    print(f"  Scores range: {np.min(scores):.4f} to {np.max(scores):.4f}")
    print(f"  Classes distribution: {np.bincount(classes.astype(np.int32))}")
    
    # Filter by confidence threshold - LOWERED to catch more objects
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

def draw_detections_cv2(image, boxes, scores, classes, class_names):
    """Draw detection boxes on the image using OpenCV with validation."""
    # Make a copy of the image to avoid modifying the original
    image_with_boxes = image.copy()
    
    # Convert to grayscale for object validation
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply edge detection to find object boundaries
    edges = cv2.Canny(gray_image, 100, 200)
    
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
    
    # Create a separate image to show box quality
    validation_image = image.copy()
    
    for i, (box, score, cls) in enumerate(zip(boxes, scores, classes)):
        # Get class name and color
        class_idx = int(cls)
        class_name = class_names[class_idx]
        color = colors[class_idx % len(colors)]
        
        # Convert normalized coordinates to pixel coordinates
        xmin, ymin, xmax, ymax = box
        xmin = max(0, int(xmin * width))
        xmax = min(width, int(xmax * width))
        ymin = max(0, int(ymin * height))
        ymax = min(height, int(ymax * height))
        
        # Skip invalid boxes
        if xmin >= xmax or ymin >= ymax:
            continue
            
        # Calculate box quality score based on edge content
        box_region = edges[ymin:ymax, xmin:xmax]
        edge_density = np.sum(box_region) / (255.0 * box_region.size) if box_region.size > 0 else 0
        
        # Adjust color based on edge density (more edges = more likely to contain an object)
        quality_indicator = min(255, int(edge_density * 1000))  # Scale up for visibility
        box_quality_color = (0, quality_indicator, 255-quality_indicator)  # From red to green
        
        # Draw rectangle with quality color
        cv2.rectangle(validation_image, (xmin, ymin), (xmax, ymax), box_quality_color, 3)
        
        # Draw rectangle with class color on main image
        cv2.rectangle(image_with_boxes, (xmin, ymin), (xmax, ymax), color, 3)
        
        # Prepare text
        label = f"{class_name}: {score:.2f}"
        quality_label = f"Quality: {edge_density:.2f}"
        
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
        
        # Add quality score text
        cv2.putText(
            validation_image,
            quality_label,
            (xmin, ymin - 5),
            font,
            font_scale,
            (255, 255, 255),
            thickness
        )
        
    # Combine the two images side by side
    if width > height:
        combined_image = np.hstack((image_with_boxes, validation_image))
    else:
        combined_image = np.vstack((image_with_boxes, validation_image))
    
    # Add a title to explain the images
    title_height = 30
    title_image = np.ones((title_height, combined_image.shape[1], 3), dtype=np.uint8) * 255
    cv2.putText(
        title_image,
        "Regular Detection | Detection Quality (Red: Low, Green: High)",
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        1
    )
    
    return np.vstack((title_image, combined_image))

def detect_objects(model, image_path, input_size=(224, 224), confidence_threshold=0.1):
    """Run object detection on a single image."""
    # Load image with OpenCV (in BGR format)
    original_image = cv2.imread(image_path)
    if original_image is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Convert to RGB for model input
    rgb_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    
    # Resize for model input
    resized_image = cv2.resize(rgb_image, input_size)
    
    # Preprocess image
    input_img = preprocess_input(resized_image.astype(np.float32))
    input_img = np.expand_dims(input_img, axis=0)  # Add batch dimension
    
    # Run inference and get raw predictions
    predictions = model.predict(input_img, verbose=0)
    
    # Debug: print raw prediction shapes and types
    print("Raw prediction types:")
    for i, pred in enumerate(predictions):
        print(f"  Prediction {i}: shape={pred.shape}, type={type(pred)}")
    
    # Extract predictions based on model output format
    if isinstance(predictions, list) and len(predictions) == 2:
        boxes, class_probs = predictions
        # Compute scores and classes from class probabilities
        scores = np.max(class_probs, axis=-1)
        classes = np.argmax(class_probs, axis=-1)
    else:
        raise ValueError(f"Unexpected model output format: {len(predictions)} elements")
    
    # Post-process detections
    boxes = boxes[0]  # Remove batch dimension
    scores = scores[0] if scores.ndim > 1 else scores
    classes = classes[0] if classes.ndim > 1 else classes
    class_probs = class_probs[0] if class_probs is not None and class_probs.ndim > 2 else class_probs
    
    # Get top-3 classes for each box if we have probability distributions
    top_classes = []
    if class_probs is not None:
        for i in range(len(boxes)):
            if i < len(class_probs):
                # Get top 3 classes and their probabilities
                probs = class_probs[i]
                top_indices = np.argsort(probs)[::-1][:3]  # Top 3 in descending order
                top_classes.append([(VOC_CLASSES[idx], float(probs[idx])) for idx in top_indices])
            else:
                top_classes.append([])
    
    # Apply thresholding and NMS with lower threshold to see more predictions
    selected_boxes, selected_scores, selected_classes = decode_predictions(
        boxes, scores, classes, confidence_threshold
    )
    
    # Also keep top classes information for selected boxes
    selected_top_classes = []
    if top_classes:
        # Get the indices of selected boxes in original boxes array
        for i, box in enumerate(selected_boxes):
            # Find matching index in original boxes
            found = False
            for j, orig_box in enumerate(boxes):
                if np.array_equal(box, orig_box):
                    selected_top_classes.append(top_classes[j])
                    found = True
                    break
            if not found:
                selected_top_classes.append([])
    
    # Draw detections on the image
    result_image = draw_detections_with_probs(
        original_image, 
        selected_boxes, 
        selected_scores, 
        selected_classes, 
        VOC_CLASSES,
        selected_top_classes
    )
    
    return result_image, selected_boxes, selected_scores, selected_classes, selected_top_classes

def draw_detections_with_probs(image, boxes, scores, classes, class_names, top_classes=None):
    """Draw detection boxes on the image with class probability information."""
    # Make a copy of the image to avoid modifying the original
    image_with_boxes = image.copy()
    
    # Convert to grayscale for object validation
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply edge detection to find object boundaries
    edges = cv2.Canny(gray_image, 100, 200)
    
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
    
    # Create a separate image for probability visualization
    probs_image = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    for i, (box, score, cls) in enumerate(zip(boxes, scores, classes)):
        # Get class name and color
        class_idx = int(cls)
        class_name = class_names[class_idx]
        color = colors[class_idx % len(colors)]
        
        # Convert normalized coordinates to pixel coordinates
        xmin, ymin, xmax, ymax = box
        xmin = max(0, int(xmin * width))
        xmax = min(width, int(xmax * width))
        ymin = max(0, int(ymin * height))
        ymax = min(height, int(ymax * height))
        
        # Skip invalid boxes
        if xmin >= xmax or ymin >= ymax:
            continue
            
        # Calculate box quality score based on edge content
        box_region = edges[ymin:ymax, xmin:xmax]
        edge_density = np.sum(box_region) / (255.0 * box_region.size) if box_region.size > 0 else 0
        
        # Draw rectangle with class color on main image
        cv2.rectangle(image_with_boxes, (xmin, ymin), (xmax, ymax), color, 3)
        
        # Prepare text
        label = f"{class_name}: {score:.2f}"
        
        # Get text size
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
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
        
        # Draw on probability image
        cv2.rectangle(probs_image, (xmin, ymin), (xmax, ymax), color, 3)
        
        y_offset = ymin + 25
        
        # Add edge density information
        cv2.putText(
            probs_image,
            f"Edge density: {edge_density:.2f}",
            (xmin + 5, y_offset),
            font,
            font_scale,
            (0, 0, 0),
            thickness
        )
        y_offset += 25
        
        # Add top class probabilities if available
        if top_classes and i < len(top_classes) and top_classes[i]:
            for j, (cls_name, prob) in enumerate(top_classes[i]):
                prob_label = f"{j+1}. {cls_name}: {prob:.3f}"
                
                # Color for probability text
                text_color = (0, 0, 200) if j == 0 else (100, 100, 100)
                
                cv2.putText(
                    probs_image,
                    prob_label,
                    (xmin + 5, y_offset),
                    font,
                    font_scale,
                    text_color,
                    thickness - 1
                )
                y_offset += 25
    
    # Combine the two images side by side
    if width > height:
        combined_image = np.hstack((image_with_boxes, probs_image))
    else:
        combined_image = np.vstack((image_with_boxes, probs_image))
    
    # Add a title to explain the images
    title_height = 30
    title_image = np.ones((title_height, combined_image.shape[1], 3), dtype=np.uint8) * 255
    cv2.putText(
        title_image,
        "Detection Results | Class Probabilities",
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        1
    )
    
    return np.vstack((title_image, combined_image))

def create_inference_model(model_path, input_size=(224, 224, 3), num_classes=21):
    """Create a new inference model that doesn't rely on Lambda layers."""
    # First, try to load the base model with unsafe deserialization enabled (for Lambda layers)
    try:
        # Enable unsafe deserialization for Lambda layers
        import keras
        original_safe_mode = keras.config.get_safe_deserialization()
        keras.config.enable_unsafe_deserialization()
        
        try:
            base_model = load_model(model_path, compile=False)
            print("Loaded model successfully with unsafe deserialization")
            # Restore original safe mode setting
            if not original_safe_mode:
                keras.config.enable_safe_deserialization()
            return base_model
        except Exception as e:
            # Restore original safe mode setting
            if not original_safe_mode:
                keras.config.enable_safe_deserialization()
            print(f"Error loading model with unsafe deserialization: {e}")
            raise e
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Creating a new inference model from scratch...")
        
        # Create a new SSD MobileNet model
        ssd_model = SSDMobileNet(input_shape=input_size, num_classes=num_classes)
        model = ssd_model.build_model()
        
        # Load weights if possible
        try:
            model.load_weights(model_path)
            print("Loaded weights successfully")
        except Exception as weight_error:
            print(f"Could not load weights: {weight_error}")
            print("Using model without pre-trained weights")
        
        # Create inference model using Keras operations instead of direct TF ops
        # This avoids the KerasTensor/TensorFlow function compatibility issue
        inference_model = ssd_model.build_inference_model()
        
        return inference_model

def main():
    parser = argparse.ArgumentParser(description='Object detection with SSD MobileNet')
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to the trained model (.keras or .h5)')
    parser.add_argument('--image-dir', type=str, required=True,
                       help='Directory containing test images')
    parser.add_argument('--output-dir', type=str, default='detection_results',
                       help='Directory to save detection results')
    parser.add_argument('--confidence-threshold', type=float, default=0.3,
                       help='Confidence threshold for detections')
    parser.add_argument('--input-size', type=int, default=300,
                       help='Input image size')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model with custom approach to handle Lambda layers
    print(f"Loading model from {args.model_path}...")
    model = create_inference_model(
        args.model_path, 
        input_size=(args.input_size, args.input_size, 3),
        num_classes=len(VOC_CLASSES)
    )
    
    # Check if image_dir exists
    if not os.path.exists(args.image_dir):
        print(f"Error: Image directory {args.image_dir} does not exist")
        return
    
    # Get all images in the directory
    image_files = [f for f in os.listdir(args.image_dir) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print(f"No image files found in {args.image_dir}")
        return
    
    # Process each image
    for i, image_file in enumerate(image_files):
        image_path = os.path.join(args.image_dir, image_file)
        print(f"Processing image {i+1}/{len(image_files)}: {image_file}")
        
        try:
            # Detect objects
            try:
                result_image, boxes, scores, classes, top_classes = detect_objects(
                    model, 
                    image_path, 
                    input_size=(args.input_size, args.input_size),
                    confidence_threshold=args.confidence_threshold
                )
            except ValueError as e:
                print(f"Error with result unpacking: {e}")
                # Fallback for older version of detect_objects function
                result_image, boxes, scores, classes = detect_objects(
                    model, 
                    image_path, 
                    input_size=(args.input_size, args.input_size),
                    confidence_threshold=args.confidence_threshold
                )
                top_classes = None
            
            # Save results
            output_path = os.path.join(args.output_dir, f"detection_{os.path.splitext(image_file)[0]}.jpg")
            cv2.imwrite(output_path, result_image)
            
            # Print detection results
            print(f"Found {len(boxes)} objects:")
            for j, (box, score, cls) in enumerate(zip(boxes, scores, classes)):
                class_name = VOC_CLASSES[int(cls)]
                print(f"  {j+1}. {class_name}: {score:.2f} at {box}")
                
            print(f"Detection saved to {output_path}")
        
        except Exception as e:
            print(f"Error processing image {image_file}: {e}")
    
    print(f"Detection results saved to {args.output_dir}")

if __name__ == '__main__':
    main()