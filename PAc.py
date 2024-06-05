from turtle import back
from typing import List, Dict
import json
import time

class Encoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        return json.JSONEncoder.default(self, obj)


class ResultSet:
    """Result set of one experiment
    
    Args:
        title: str.  Name of the result set.
        filename: str. Name of the file to store the results in. Should omit the 
            extension.  If no filename is given, one will be generated from the
            title

    
    """
    def __init__(self, title: str, filename: str|None = None):
        self.title = title
        if (filename):
            self.filename = filename if filename.endswith(".json") else (filename + ".json")
        else:
            self.filename = title.lower().replace(" ","_") + ".json"
        self.x: list[float] = []
        self.y: list[float] = []
        self.labels: list[str] = []
        self.last_saved_on = None
        
    def save(self):
        self.last_saved_on = time.time()
        with open(f"{self.filename}.json", "w") as f:
            json.dump(self.__dict__, f, cls=Encoder)
            
    @staticmethod
    def load(name: str):
        with open(f"{name}.json", "r") as f:
            data = json.load(f)
            result = ResultSet(data['title'], data['filename'])
            result.x = data['x']
            result.y = data['y']
            result.labels = data['labels']
            result.last_saved_on = data['last_saved_on']
            return result
        
    def add(self, label: str, x: List[float], y: List[float]):
        self.x.extend(x)
        self.y.extend(y)
        self.labels.append(label)
        
    def get(self, label: str) -> Dict[str, List[float]]:
        return {'x': [self.x[self.labels.index(label)]], 'y': [self.y[self.labels.index(label)]]}



import numpy as np
import tensorflow as tf
from keras.src import activations, backend
from keras.src.api_export import keras_export
from keras.src.layers import Layer
from typing import Callable, Tuple

@keras_export('keras.layers.PAc')
class PAc(Layer):
    """Pre-computed Activation Function Layer.
    
    Args:
        func: Callable, the activation function to be used to initialize the lookup table.
        x_low: float, the lower bound of the lookup table.
        x_high: float, the upper bound of the lookup table.
        func_low: Callable, the activation function to be used for x <= x_low.
        func_high: Callable, the activation function to be used for x >= x_high.
        n: int, the number of points in the lookup table.
            Default: 64

    Formula:
    ``` python
    f(x) = func_low(x) if x <= x_low
    f(x) = func_high(x) if x >= x_high
    f(x) is looked up in the computed lookup table if x_low < x < x_high
    ```
    """
    
    def __init__(self, func: Callable, x_low: float, x_high: float, func_low: Callable, func_high: Callable, n: int = 64, **kwargs):
        super().__init__(trainable=False, **kwargs)
        self.table = [func(x_low + (i + 0.5) * (x_high - x_low) / n) for i in range(n)]
        self.x_low = x_low
        self.x_high = x_high
        self.func_low = func_low
        self.func_high = func_high
        self.n = n
        self.mult = n / (x_high - x_low)
        self.add = (x_low * n) / (x_low - x_high)

    def build(self, input_shape):
        pass

    def call(self, inputs: tf.Tensor|List[float]):
        if (isinstance(inputs, tf.Tensor)):
            return tf.map_fn(self.pac, inputs)
        else:
            return tf.map_fn(self.pac, backend.convert_to_tensor(inputs))
    
    def pac(self, x: float)->float:
        if x <= self.x_low:
            return self.func_low(x)
        elif x >= self.x_high:
            return self.func_high(x)
        else:
            idx = np.floor(x * self.mult + self.add).astype(int)
            return self.table[idx]
        
    def get_config(self):
        config = super().get_config()
        config.update({
            'x_low': self.x_low,
            'x_high': self.x_high,
            'func_low': self.func_low,
            'func_high': self.func_high,
            'n': self.n,
            'table': self.table
        })
        return config
    
    def compute_output_shape(self, input_shape):
        return input_shape
    
    def get_xy(self) -> Tuple[List[float], List[float]]:
        return np.linspace(self.x_low, self.x_high, self.n).tolist(), self.table

