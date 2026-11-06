import numpy as np


class XV_Test:
    def __init__(self):
        self.data = np.arange(100)
        self.train_data = self.data[:80]
        self.test_data = self.data[80:]
        self.leaked_test = self.data[79:-1]
