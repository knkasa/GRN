import tensorflow as tf
from tensorflow.keras import layers

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

batch_size = 32
input_dim = 8
context_dim = 4
hidden_units = 16

# Sample inputs
x = tf.random.normal((batch_size, input_dim))
context = tf.random.normal((batch_size, context_dim))

# With context
grn = GatedResidualNetwork(hidden_units=hidden_units, use_context=True)
output = grn(x, context=context)

# Without context
grn_no_context = GatedResidualNetwork(hidden_units=hidden_units, use_context=False)
output_no_context = grn_no_context(x)

print(output.shape)  # (32, 16)
