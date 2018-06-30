import tensorflow as tf 
from tensorflow.contrib.data import Dataset. Iterator

class ReplayTable:

    def __init__(self, max_records=50000, batch_size=8):
        self.max_records = 50000