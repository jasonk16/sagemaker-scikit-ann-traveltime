import os
import json
import torch
import torch.nn as nn
import torch.functional as f
import numpy as np

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim=1):
        super(MLP, self).__init__()

        assert out_dim == 1, 'out_dim must be 1'

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.hidden_layers = nn.ModuleList([nn.Linear(self.in_dim, self.hidden_dim[0])])

        hidden_layer_sizes = zip(self.hidden_dim[:-1], self.hidden_dim[1:])
        for (layer_in, layer_out) in hidden_layer_sizes:
            self.hidden_layers.extend([nn.Linear(layer_in, layer_out)])

        self.output = nn.Linear(self.hidden_dim[-1], self.out_dim)

    def forward(self, x):

        for h_layer in iter(self.hidden_layers):
            x = torch.relu(h_layer(x))

        x = self.output(x)

        x = x.squeeze(1)
        return x

def model_fn(model_dir):

    print("Loading the model")

    mlp_model = MLP(6, [16,18,20,14,10,6], 1)

    with open(os.path.join(model_dir, 'mlp_timePrediction_pytorch.pt'), 'rb') as f:
        mlp_model.load_state_dict(torch.load(f))

    mlp_model.eval()
    return mlp_model

def input_fn(request_body, content_type='application/json'):
    print('Getting input data: ')
    if content_type == 'application/json':
        input_data = json.loads(request_body)
        print('input data: ', input_data)
        input_values = np.array([[input_data['hod'], input_data['origin_long'], input_data['origin_lat'], input_data['dest_long'], input_data['dest_lat'],input_data['distance']]], dtype = np.float).tolist()
        print("Input Values: ", input_values, type(input_values))
        input_all = torch.tensor(input_values, dtype=torch.float )
        return input_all

    raise Exception(f'Requested unsupported ContentType in content_type: {content_type}')

def output_fn(prediction_output, accept='application/json'):
    print("Generating output")
    if accept == 'application/json':
        print(prediction_output[0], type(prediction_output))
        prediction_output = json.dumps(str(prediction_output[0]))
        print("PRED OUT ", prediction_output)
        return prediction_output, accept
    raise Exception(f'Requested unsupported ContentType in Accept: {accept}')

def predict_fn(input_all, mlp_model):
    print("Generating prediction")
    print(input_all)
    predicted_time = mlp_model(input_all)
    y_pred = predicted_time.data.cpu().numpy()
    print(y_pred)
    return y_pred

