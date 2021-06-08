from tensorflow.keras.layers import Layer
from tensorflow.keras.activations import softmax

class Softmax(Layer):
    def __init__(self, axis):
        super(Softmax, self).__init__()
        self.axis = axis

    def call(self, inputs):
        return softmax(inputs, axis=self.axis)
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'axis': self.axis 
        })
        return config