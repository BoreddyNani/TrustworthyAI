import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2

class SSDMobileNet:
    """SSD MobileNet implementation for object detection."""
    
    def __init__(self, input_shape=(224, 224, 3), num_classes=21):
        self.input_shape = input_shape
        self.num_classes = num_classes
        # Default anchor boxes - better configured for PASCAL VOC dataset
        self.anchor_scales = [0.1, 0.2, 0.37, 0.54, 0.71, 0.88]
        self.anchor_ratios = [0.5, 1.0, 2.0]
        
    def build_model(self):
        """Build the SSD MobileNet model architecture."""
        # Input layer
        input_layer = layers.Input(shape=self.input_shape)
        
        # Use MobileNetV2 as base network (backbone) with fewer layers frozen
        base_model = MobileNetV2(
            input_shape=self.input_shape,
            include_top=False,
            weights='imagenet'
        )
        
        # Only freeze the first few layers - allow more layers to be trained
        for layer in base_model.layers[:50]:  # Freeze only first 50 layers instead of all
            layer.trainable = False
        
        # Get the base network output
        x = base_model(input_layer)
        
        # Create a better SSD model with more feature maps and improved prediction heads
        # Feature map 1 (from base model)
        feature_map1 = layers.Conv2D(512, 3, padding='same', activation='relu', name='feature_map1')(x)
        
        # Feature map 2
        feature_map2 = layers.Conv2D(512, 3, strides=2, padding='same', activation='relu', name='feature_map2')(feature_map1)
        
        # Feature map 3
        feature_map3 = layers.Conv2D(256, 3, strides=2, padding='same', activation='relu', name='feature_map3')(feature_map2)
        
        # Feature map 4 - additional feature map for better multi-scale detection
        feature_map4 = layers.Conv2D(256, 3, strides=2, padding='same', activation='relu', name='feature_map4')(feature_map3)
        
        # Use these feature maps for detection
        feature_maps = [feature_map1, feature_map2, feature_map3, feature_map4]
        
        # Create prediction heads for each feature map
        box_outputs = []
        class_outputs = []
        
        for i, feature_map in enumerate(feature_maps):
            # Box predictions with batch normalization for better training
            box_output = layers.Conv2D(16, 3, padding='same')(feature_map)
            box_output = layers.BatchNormalization()(box_output)
            box_output = layers.Activation('relu')(box_output)
            box_output = layers.Flatten()(box_output)
            box_output = layers.Dense(128)(box_output)
            box_output = layers.BatchNormalization()(box_output)
            box_output = layers.Activation('relu')(box_output)
            box_outputs.append(box_output)
            
            # Class predictions with dropout to prevent overfitting
            class_output = layers.Conv2D(self.num_classes * 4, 3, padding='same')(feature_map)
            class_output = layers.BatchNormalization()(class_output)
            class_output = layers.Activation('relu')(class_output)
            class_output = layers.Flatten()(class_output)
            class_output = layers.Dense(256)(class_output)
            class_output = layers.Dropout(0.3)(class_output)  # Add dropout to prevent overfitting
            class_output = layers.BatchNormalization()(class_output)
            class_output = layers.Activation('relu')(class_output)
            class_outputs.append(class_output)
        
        # Concatenate the outputs
        box_output = layers.Concatenate()(box_outputs)
        class_output = layers.Concatenate()(class_outputs)
        
        # Final output layers
        box_output = layers.Dense(256)(box_output)
        box_output = layers.Dense(80)(box_output)  # Maximum 20 boxes with 4 coordinates each
        
        class_output = layers.Dense(512)(class_output)
        class_output = layers.Dense(20 * self.num_classes)(class_output)
        class_output = layers.Reshape((20, self.num_classes))(class_output)
        
        # Use sigmoid activation for background and softmax for the rest to better separate background
        # This helps prevent the model from being biased toward background class
        background_probs = layers.Lambda(lambda x: tf.nn.sigmoid(x[:, :, 0:1]))(class_output)
        object_probs = layers.Lambda(lambda x: tf.nn.softmax(x[:, :, 1:]))(class_output)
        class_output = layers.Concatenate(axis=-1)([background_probs, object_probs])
        
        # Reshape box output to match expected format
        box_output = layers.Reshape((20, 4))(box_output)
        
        # Create model
        model = Model(inputs=input_layer, outputs=[box_output, class_output], name='ssd_mobilenet_improved')
        
        return model
    
    def build_inference_model(self):
        """Build a model for inference with additional post-processing."""
        # Get the base model
        model = self.build_model()
        
        # Simply pass through the raw boxes and class probabilities
        # This avoids KerasTensor/TF function compatibility issues
        boxes, classes = model.outputs
        
        # Create the inference model that returns raw outputs
        # We'll handle post-processing in Python code
        inference_model = Model(
            inputs=model.inputs,
            outputs=[boxes, classes]  # Return raw boxes and class probabilities
        )
        
        return inference_model
    
    def ssd_loss(self, alpha=1.0, gamma=2.0):
        """Custom SSD loss function with focal loss to address class imbalance."""
        @tf.function  # Use tf.function to optimize the loss calculation
        def compute_loss(y_true, y_pred):
            # Access tensors by index instead of unpacking
            y_pred_boxes = y_pred[0]    # First element is box predictions
            y_pred_classes = y_pred[1]  # Second element is class predictions
            
            y_true_boxes = y_true[0]    # First element is true boxes
            y_true_classes = y_true[1]  # Second element is true classes
            
            # Create a mask for valid boxes (non-zero)
            valid_mask = tf.reduce_sum(y_true_boxes, axis=-1) > 0
            
            # Localization loss (smooth L1 loss) - only for valid boxes
            valid_true_boxes = tf.boolean_mask(y_true_boxes, valid_mask)
            valid_pred_boxes = tf.boolean_mask(y_pred_boxes, valid_mask)
            
            # Use Huber loss for bounding box regression
            loc_loss = tf.keras.losses.Huber()(valid_true_boxes, valid_pred_boxes)
            
            # Class weights to balance background vs. object classes
            # Higher weight for non-background classes (indices > 0)
            background_weight = 1.0
            object_weight = 5.0  # Give 5x weight to actual object classes
            
            # Create a weight tensor based on class labels
            class_weights = tf.where(
                tf.argmax(y_true_classes, axis=-1) > 0,
                object_weight,  # Weight for non-background classes
                background_weight  # Weight for background class
            )
            
            # Implement focal loss to handle class imbalance
            # Focal loss: -alpha * (1-p)^gamma * log(p) where p is the predicted probability for true class
            # This reduces the influence of easy examples (like background) and focuses on hard ones
            
            # Get predicted probability for the true class
            true_class_probs = tf.reduce_sum(y_true_classes * y_pred_classes, axis=-1)
            
            # Calculate focal weights
            focal_weights = tf.pow(1.0 - true_class_probs, gamma)
            
            # Apply focal loss with categorical crossentropy
            class_loss = tf.keras.losses.CategoricalCrossentropy(
                reduction=tf.keras.losses.Reduction.NONE
            )(y_true_classes, y_pred_classes)
            
            # Apply class weights and focal weights
            weighted_class_loss = class_loss * class_weights * focal_weights
            
            # Take mean of weighted loss
            class_loss = tf.reduce_mean(weighted_class_loss)
            
            # Total loss is weighted sum - give more weight to classification to focus on correct class prediction
            total_loss = loc_loss + 2.0 * class_loss  # Increase classification loss weight
            
            return total_loss
        
        return compute_loss