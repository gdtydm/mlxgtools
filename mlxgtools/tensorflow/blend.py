import tensorflow as tf 

def avg_blend_models(input_shape, models: list):
    inputs = tf.keras.layers.Input(shape=input_shape) # your input shape
    predictions = [m(inputs) for m in models]
    outputs = tf.keras.layers.Average()(predictions) # whatever aggregation you want
    return tf.keras.Model(inputs, outputs)