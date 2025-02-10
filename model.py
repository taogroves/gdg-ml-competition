import torch
import torch.nn as nn
import torch.optim as optim


# Define model (feed-forward, two hidden layers)
# TODO: This is where most of the work will be done. You can change the model architecture,
#       add more layers, change activation functions, etc.
class MyModel(nn.Module):
    def __init__(self, input_dim):
        super(MyModel, self).__init__()
        
        self.layer1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)  # Add batch normalization
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)
        self.layer2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)  # Add batch normalization
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)

        self.layer3 = nn.Linear(128, 64)  # Moved previous layer
        self.bn3 = nn.BatchNorm1d(64)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(0.2)

        self.output_layer = nn.Linear(64, 1)

    def forward(self, x):
        # Run data through 3 layers and apply normalization, activation, and dropout
        x = self.layer1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.layer2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.layer3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.dropout3(x)
        # Finally, push through output layer to get predictions
        x = self.output_layer(x)
        return x

def create_model(features):
    model = MyModel(features.shape[1])

    # define optimizer (feel free to change this)
    # RMSprop is a good optimizer that adapts learning rates for each parameter
    optimizer = optim.RMSprop(model.parameters(), lr=0.0005)

    return model, optimizer

if __name__ == '__main__':
    # create sample model with 228 input features
    model, _ = create_model(torch.zeros(1, 228))
    print(model)

    