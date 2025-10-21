# Encoding

import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder

data = np.array([['red'],
                 ['green'],
                 ['blue']])

print(data)

# Ordinal Encoding
"""
encoder = OrdinalEncoder()

result = encoder.fit_transform(data)

print(result)
"""


# One Hot Encoding
encoder = OneHotEncoder(sparse_output=False)

result = encoder.fit_transform(data)

print(result)
