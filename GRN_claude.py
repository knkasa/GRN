import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

class GatedResidualBlock(layers.Layer):
    """
    Basic Gated Residual Block
    Implements: output = gate * residual + (1 - gate) * transformed_input
    """
    def __init__(self, units, activation='relu', **kwargs):
        super(GatedResidualBlock, self).__init__(**kwargs)
        self.units = units
        self.activation = activation
        
        # Main transformation layers
        self.dense1 = layers.Dense(units, activation=activation)
        self.dense2 = layers.Dense(units, activation=activation)
        
        # Gate computation
        self.gate_dense = layers.Dense(units, activation='sigmoid')
        
        # Projection layer for dimension matching (if needed)
        self.projection = None
        
    def build(self, input_shape):
        # Add projection layer if input dimension doesn't match output
        if input_shape[-1] != self.units:
            self.projection = layers.Dense(self.units)
        super(GatedResidualBlock, self).build(input_shape)
    
    def call(self, inputs, training=None):
        # Residual connection (with projection if needed)
        residual = inputs
        if self.projection is not None:
            residual = self.projection(residual)
        
        # Main transformation path
        x = self.dense1(inputs)
        x = self.dense2(x)
        
        # Gate computation
        gate = self.gate_dense(inputs)
        
        # Gated residual connection
        output = gate * residual + (1 - gate) * x
        
        return output

class AdvancedGatedResidualBlock(layers.Layer):
    """
    Advanced Gated Residual Block with batch normalization and dropout
    """
    def __init__(self, units, activation='relu', dropout_rate=0.1, use_batch_norm=True, **kwargs):
        super(AdvancedGatedResidualBlock, self).__init__(**kwargs)
        self.units = units
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        
        # Main transformation layers
        self.dense1 = layers.Dense(units)
        self.dense2 = layers.Dense(units)
        
        # Batch normalization layers
        if use_batch_norm:
            self.bn1 = layers.BatchNormalization()
            self.bn2 = layers.BatchNormalization()
        
        # Activation layers
        self.act1 = layers.Activation(activation)
        self.act2 = layers.Activation(activation)
        
        # Dropout layers
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)
        
        # Gate computation with its own normalization
        self.gate_dense = layers.Dense(units)
        self.gate_bn = layers.BatchNormalization() if use_batch_norm else None
        self.gate_activation = layers.Activation('sigmoid')
        
        # Projection layer
        self.projection = None
        self.proj_bn = None
        
    def build(self, input_shape):
        if input_shape[-1] != self.units:
            self.projection = layers.Dense(self.units)
            if self.use_batch_norm:
                self.proj_bn = layers.BatchNormalization()
        super(AdvancedGatedResidualBlock, self).build(input_shape)
    
    def call(self, inputs, training=None):
        # Residual connection with projection if needed
        residual = inputs
        if self.projection is not None:
            residual = self.projection(residual)
            if self.proj_bn is not None:
                residual = self.proj_bn(residual, training=training)
        
        # Main transformation path
        x = self.dense1(inputs)
        if self.use_batch_norm:
            x = self.bn1(x, training=training)
        x = self.act1(x)
        x = self.dropout1(x, training=training)
        
        x = self.dense2(x)
        if self.use_batch_norm:
            x = self.bn2(x, training=training)
        x = self.act2(x)
        x = self.dropout2(x, training=training)
        
        # Gate computation
        gate = self.gate_dense(inputs)
        if self.gate_bn is not None:
            gate = self.gate_bn(gate, training=training)
        gate = self.gate_activation(gate)
        
        # Gated residual connection
        output = gate * residual + (1 - gate) * x
        
        return output

class GatedResidualNetwork(keras.Model):
    """
    Complete Gated Residual Network
    """
    def __init__(self, layer_sizes, num_classes, use_advanced=True, **kwargs):
        super(GatedResidualNetwork, self).__init__(**kwargs)
        self.layer_sizes = layer_sizes
        self.num_classes = num_classes
        
        # Input layer
        self.input_layer = layers.Dense(layer_sizes[0], activation='relu')
        
        # Gated residual blocks
        self.gated_blocks = []
        BlockClass = AdvancedGatedResidualBlock if use_advanced else GatedResidualBlock
        
        for size in layer_sizes[1:]:
            self.gated_blocks.append(BlockClass(size))
        
        # Output layer
        if num_classes > 1:
            self.output_layer = layers.Dense(num_classes, activation='softmax')
        else:
            self.output_layer = layers.Dense(1, activation='sigmoid')
    
    def call(self, inputs, training=None):
        x = self.input_layer(inputs)
        
        for block in self.gated_blocks:
            x = block(x, training=training)
        
        output = self.output_layer(x)
        return output

# Example usage and training
def create_sample_data(n_samples=1000, n_features=20, n_classes=3):
    """Create sample classification data"""
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, n_classes, n_samples)
    return X.astype(np.float32), tf.keras.utils.to_categorical(y, n_classes)

def train_gated_resnet():
    """Example training function"""
    # Create sample data
    X_train, y_train = create_sample_data(1000, 20, 3)
    X_test, y_test = create_sample_data(200, 20, 3)
    
    # Create model
    model = GatedResidualNetwork(
        layer_sizes=[64, 128, 64, 32],
        num_classes=3,
        use_advanced=True
    )
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=50,
        batch_size=32,
        verbose=1
    )
    
    return model, history

# Functional API version for more complex architectures
def create_gated_resnet_functional(input_shape, layer_sizes, num_classes):
    """
    Functional API implementation for more flexibility
    """
    inputs = keras.Input(shape=input_shape)
    
    # Initial dense layer
    x = layers.Dense(layer_sizes[0], activation='relu')(inputs)
    
    # Add gated residual blocks
    for size in layer_sizes[1:]:
        # Store input for residual connection
        residual = x
        
        # Main transformation
        transformed = layers.Dense(size, activation='relu')(x)
        transformed = layers.Dense(size, activation='relu')(transformed)
        
        # Gate computation
        gate = layers.Dense(size, activation='sigmoid')(x)
        
        # Projection if dimensions don't match
        if x.shape[-1] != size:
            residual = layers.Dense(size)(residual)
        
        # Gated residual connection
        x = layers.Lambda(lambda inputs: inputs[0] * inputs[1] + (1 - inputs[0]) * inputs[2])([gate, residual, transformed])
    
    # Output layer
    if num_classes > 1:
        outputs = layers.Dense(num_classes, activation='softmax')(x)
    else:
        outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

# Example with custom gate function
class CustomGatedResidualBlock(layers.Layer):
    """
    Gated residual block with custom gate computation
    """
    def __init__(self, units, gate_type='sigmoid', **kwargs):
        super(CustomGatedResidualBlock, self).__init__(**kwargs)
        self.units = units
        self.gate_type = gate_type
        
        self.dense1 = layers.Dense(units, activation='relu')
        self.dense2 = layers.Dense(units, activation='relu')
        self.gate_dense = layers.Dense(units)
        
        # Different gate activation functions
        if gate_type == 'sigmoid':
            self.gate_activation = layers.Activation('sigmoid')
        elif gate_type == 'tanh':
            self.gate_activation = layers.Activation('tanh')
        elif gate_type == 'softmax':
            self.gate_activation = layers.Softmax(axis=-1)
        
        self.projection = None
        
    def build(self, input_shape):
        if input_shape[-1] != self.units:
            self.projection = layers.Dense(self.units)
        super(CustomGatedResidualBlock, self).build(input_shape)
    
    def call(self, inputs, training=None):
        residual = inputs
        if self.projection is not None:
            residual = self.projection(residual)
        
        # Main path
        x = self.dense1(inputs)
        x = self.dense2(x)
        
        # Gate computation
        gate = self.gate_dense(inputs)
        gate = self.gate_activation(gate)
        
        # Apply gating
        if self.gate_type == 'tanh':
            # For tanh, we need to adjust the formula
            gate = (gate + 1) / 2  # Convert from [-1,1] to [0,1]
        
        output = gate * residual + (1 - gate) * x
        return output

if __name__ == "__main__":
    # Example usage
    print("Training Gated Residual Network...")
    model, history = train_gated_resnet()
    print(f"Final validation accuracy: {max(history.history['val_accuracy']):.4f}")
    
    # Create functional model example
    print("\nCreating functional model...")
    func_model = create_gated_resnet_functional(
        input_shape=(20,),
        layer_sizes=[64, 128, 64, 32],
        num_classes=3
    )
    func_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print(func_model.summary())