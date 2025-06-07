import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

# Gated Residual Network (work for table)
# https://medium.com/chat-gpt-now-writes-all-my-articles/gated-residual-networks-a-modern-secret-weapon-for-tabular-deep-learning-7a8d247a01d1


class GatedResidualNetwork(tf.keras.layers.Layer):
    def __init__(self, hidden_units, dropout_rate=0.0, use_context=False, **kwargs):
        super(GatedResidualNetwork, self).__init__(**kwargs)
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate
        self.use_context = use_context

        self.dense1 = layers.Dense(hidden_units)
        self.elu = layers.ELU()
        self.dense2 = layers.Dense(hidden_units)
        self.dropout = layers.Dropout(dropout_rate)
        self.gate_dense = layers.Dense(hidden_units)
        self.sigmoid = layers.Activation("sigmoid")
        self.layer_norm = layers.LayerNormalization()
        self.output_layer = layers.Dense(1)  # assuming regression

        if self.use_context:
            self.context_dense = layers.Dense(hidden_units)

        self.skip_dense = None  # Will be defined in build()

    # build is called when you run 
    def build(self, input_shape):
        super(GatedResidualNetwork, self).build(input_shape) # may or may not needed here.

        # If input contains context data, use if statement below to get input_dim.
        if isinstance(input_shape, (list, tuple)):
            input_dim = input_shape[0][-1]
        else:
            input_dim = input_shape[-1]

        if input_dim != self.hidden_units:
            self.skip_dense = layers.Dense(self.hidden_units)

    def call(self, x, context=None, training=False):
        
        if self.use_context and context is not None:
            x_context = tf.concat([x, context], axis=-1)
        else:
            x_context = x

        #------------ Residual block ------------------------
        x1 = self.dense1(x_context)
        x1 = self.elu(x1)
        x1 = self.dense2(x1)
        x1 = self.dropout(x1, training=training)

        gate = self.sigmoid(self.gate_dense(x1))
        x_gated = x1*gate

        if self.skip_dense is not None:
            residual = self.skip_dense(x)
        else:
            residual = x

        out = self.layer_norm(x_gated + residual)
        #-----------------------------------------------------

        out = self.output_layer(out)  # assuming regression

        return out


np.random.seed(42)
tf.random.set_seed(42)

batch_size = 32
input_dim = 8
context_dim = 4
hidden_units = 16

def generate_data(n_samples=1000, input_dim=8):
    X = np.random.normal(size=(n_samples, input_dim)).astype(np.float32)
    y = (X[:, 0] * 2 - X[:, 1]**2 + np.sin(X[:, 2] * np.pi)).astype(np.float32)
    return X, y

X_train, y_train = generate_data(1000)
X_test, y_test = generate_data(200)

def build_model(input_dim, hidden_units):
    inputs = tf.keras.Input(shape=(input_dim,))
    outputs= GatedResidualNetwork(hidden_units)(inputs)
    model = models.Model(inputs, outputs)
    return model

model = build_model(input_dim, hidden_units)
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

history = model.fit(X_train, y_train, 
                    epochs=50, 
                    batch_size=32,
                    validation_split=0.2, 
                    verbose=1
                    )

loss, mae = model.evaluate(X_test, y_test)
print(f"Test MAE: {mae:.3f}")
