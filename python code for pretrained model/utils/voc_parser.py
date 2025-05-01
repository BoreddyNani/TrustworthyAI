import os
import numpy as np
import xml.etree.ElementTree as ET
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.utils import load_img, img_to_array

# PASCAL VOC dataset classes
VOC_CLASSES = [
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

# Class mapping: name to index
CLASS_MAPPING = {class_name: i for i, class_name in enumerate(VOC_CLASSES)}

def parse_annotation(annotation_path):
    """Parse a PASCAL VOC annotation file."""
    tree = ET.parse(annotation_path)
    root = tree.getroot()
    
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    
    objects = []
    for obj in root.findall('object'):
        class_name = obj.find('name').text
        if class_name not in CLASS_MAPPING:
            continue
            
        bbox = obj.find('bndbox')
        xmin = float(bbox.find('xmin').text) / width
        ymin = float(bbox.find('ymin').text) / height
        xmax = float(bbox.find('xmax').text) / width
        ymax = float(bbox.find('ymax').text) / height
        
        # Skip invalid boxes
        if xmin >= xmax or ymin >= ymax:
            continue
            
        objects.append({
            'class': CLASS_MAPPING[class_name],
            'box': [xmin, ymin, xmax, ymax]
        })
    
    return objects, width, height

def load_dataset(data_dir, split='train'):
    """Load PASCAL VOC dataset for a specific split."""
    images_dir = os.path.join(data_dir, 'VOCdevkit', 'VOC2007', 'JPEGImages')
    annots_dir = os.path.join(data_dir, 'VOCdevkit', 'VOC2007', 'Annotations')
    split_file = os.path.join(data_dir, 'VOCdevkit', 'VOC2007', 'ImageSets', 'Main', f'{split}.txt')
    
    with open(split_file, 'r') as f:
        image_ids = [line.strip() for line in f.readlines()]
    
    dataset = []
    for image_id in image_ids:
        image_path = os.path.join(images_dir, f'{image_id}.jpg')
        annot_path = os.path.join(annots_dir, f'{image_id}.xml')
        
        if not os.path.exists(image_path) or not os.path.exists(annot_path):
            continue
            
        objects, width, height = parse_annotation(annot_path)
        if len(objects) == 0:
            continue
            
        dataset.append({
            'image_path': image_path,
            'objects': objects,
            'width': width,
            'height': height
        })
    
    print(f"Loaded {len(dataset)} images for {split} split")
    return dataset

def pad_to_fixed_size(boxes, labels, max_boxes=20):
    """Pad boxes and labels to a fixed size."""
    padded_boxes = np.zeros((max_boxes, 4), dtype=np.float32)
    padded_labels = np.zeros(max_boxes, dtype=np.int32)
    
    # Fill in the actual values
    if len(boxes) > 0:
        num_boxes = min(len(boxes), max_boxes)
        padded_boxes[:num_boxes] = boxes[:num_boxes]
        padded_labels[:num_boxes] = labels[:num_boxes]
    
    return padded_boxes, padded_labels

def process_sample(sample, input_size=(224, 224), max_boxes=20):
    """Process a single sample for training."""
    # Load and preprocess image
    image = load_img(sample['image_path'], target_size=input_size)
    image = img_to_array(image)
    image = preprocess_input(image)
    
    # Process boxes and labels
    boxes = np.array([obj['box'] for obj in sample['objects']], dtype=np.float32)
    labels = np.array([obj['class'] for obj in sample['objects']], dtype=np.int32)
    
    # Pad boxes and labels to fixed size
    padded_boxes, padded_labels = pad_to_fixed_size(boxes, labels, max_boxes)
    
    # Convert labels to one-hot encoding
    one_hot_labels = tf.one_hot(padded_labels, depth=len(VOC_CLASSES))
    
    # Return the processed sample
    return image, (padded_boxes, one_hot_labels)

def create_tf_dataset(dataset, input_size=(224, 224), batch_size=8, max_boxes=20, 
                      shuffle=True, augment=False, prefetch=True, repeat=True):
    """Create a TensorFlow dataset from the VOC dataset."""
    # Create a dataset of sample indices
    indices = list(range(len(dataset)))
    
    # Define generator function
    def generator():
        if shuffle:
            np.random.shuffle(indices)
            
        for idx in indices:
            sample = dataset[idx]
            yield process_sample(sample, input_size, max_boxes)
    
    # Define output signature
    output_signature = (
        tf.TensorSpec(shape=(input_size[0], input_size[1], 3), dtype=tf.float32),
        (
            tf.TensorSpec(shape=(max_boxes, 4), dtype=tf.float32),
            tf.TensorSpec(shape=(max_boxes, len(VOC_CLASSES)), dtype=tf.float32)
        )
    )
    
    # Create TensorFlow dataset
    tf_dataset = tf.data.Dataset.from_generator(
        generator, 
        output_signature=output_signature
    )
    
    # Shuffle the dataset if needed
    if shuffle:
        tf_dataset = tf_dataset.shuffle(buffer_size=min(1000, len(dataset)))
    
    # Batch the dataset
    tf_dataset = tf_dataset.batch(batch_size)
    
    # Add repeat for multiple epochs
    if repeat:
        tf_dataset = tf_dataset.repeat()
    
    # Prefetch for better performance
    if prefetch:
        tf_dataset = tf_dataset.prefetch(tf.data.AUTOTUNE)
    
    return tf_dataset