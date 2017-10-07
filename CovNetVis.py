import argparse
from keras.models import model_from_json

# the name of the file containg model.
# x.json.. x.h5 must must the weights
model_file = 'model4.json'

with open(model_file, 'r') as jfile:
    # NOTE: if you saved the file by calling json.dump(model.to_json(), ...)
    # then you will have to call:
    #
    # model = model_from_json(json.loads(jfile.read()))
    #
    # instead.
    model = model_from_json(jfile.read())

model.compile("adam", "mse")
weights_file = model_file.replace('json', 'h5')
model.load_weights(weights_file)
model.summary()


"""Model loaded"""


from vis.visualization import visualize_activation
from vis.utils import utils
from keras import activations


layer_idx = utils.find_layer_idx(model, 'preds')

# Swap softmax with linear
model.layers[layer_idx].activation = activations.linear
model = utils.apply_modifications(model)

# This is the output node we want to maximize.
filter_idx = 0
img = visualize_activation(model, layer_idx, filter_indices=filter_idx)
plt.imshow(img[..., 0])



