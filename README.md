# Supervised Learning CNN 

[Simple classification scenario](https://keras.io/api/datasets/fashion_mnist) to train a model using TensorFlow and Keras.
This project provides a template to build a supervised model.
## Installation
Please make sure, you have installed [Python](https://realpython.com/installing-python/).
Use the package manager [pip](https://pip.pypa.io/en/stable/) .

```bash
# Requires the latest pip
pip install --upgrade pip

# Requires tensorflow
pip install tensorflow
```

## Usage
Try different [activation functions](https://keras.io/api/layers/activations/) and other parameters such as the learning rate or techniques like [Dropout](https://jmlr.org/papers/v15/srivastava14a.html).
```python
# activation function
fnActivation = 'relu'
# epochs
epochs = 5
# learning rate
lr = 0.0001
# dropout 
model.add(layers.Dropout(0.2))
```
## Evaluation 
Test accuracy: 0.8970999717712402