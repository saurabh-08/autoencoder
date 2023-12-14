# importing libraries

import torch
import torch.optim as optim
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
from torchvision import datasets, transforms


# hyperparameters
batch_size = 128
learning_rate = 0.001
epochs = 10

transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,))])

# mnist dataset import
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Fully Connected Layers Autoencoder
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(28*28, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU())
        self.decoder = nn.Sequential(nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 28*28), nn.Tanh())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Convolutional Autoencoder
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        self.encoder = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2), nn.Conv2d(16, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2))
        self.decoder = nn.Sequential(nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.Conv2d(16, 1, 3, padding=1), nn.Tanh(), nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# model training
def train_model(model, train_loader):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    for epoch in range(epochs):
        for data, _ in train_loader:
            if isinstance(model, Autoencoder):
                data = data.view(data.size(0), -1)
            else:
                data = data
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, data)
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}') #printing output

# visualization plots function
def visualize_outputs(model, test_loader):
    model.eval()
    selected_img = [[] for _ in range(10)]
    with torch.no_grad():
        for images, labels in test_loader:
            for img, label in zip(images, labels):
                if len(selected_img[label.item()]) < 2:
                    selected_img[label.item()].append(img)
                if all(len(imgs) == 2 for imgs in selected_img):
                    break
            if all(len(imgs) == 2 for imgs in selected_img):
                break

    fig, axes = plt.subplots(nrows=2, ncols=20, figsize=(20, 4))
    for i, imgs in enumerate(selected_img):
        for j, img in enumerate(imgs):
            if isinstance(model, Autoencoder):
                img = img.view(-1, 28*28)
            else:
                img = img.unsqueeze(0)
            output = model(img)

            # displaying original image
            ax = axes[0, i * 2 + j]
            original_img = img[0].reshape(28, 28).detach().numpy()
            ax.imshow(original_img, cmap='gray')
            ax.axis('off')
            ax.set_title(str(i))

            # displaying reconstructed image
            ax = axes[1, i * 2 + j]
            if isinstance(model, ConvAutoencoder):
              output_img = output[0].reshape(28, 28).detach().numpy()
            else:
              output_img = output.reshape(28, 28).detach().numpy()
            ax.imshow(output_img, cmap='gray')
            ax.axis('off')

    plt.show()


# Count parameters in encoder & decoder
def count_encoder_decoder_parameters(model):
    encoder_params = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
    decoder_params = sum(p.numel() for p in model.decoder.parameters() if p.requires_grad)
    return encoder_params, decoder_params

# Fully Connected Autoencoder
model_fc = Autoencoder()
train_model(model_fc, train_loader)
encoder_params_fc, decoder_params_fc = count_encoder_decoder_parameters(model_fc)
print(f"Fully Connected Autoencoder - Encoder Parameters: {encoder_params_fc}, Decoder Parameters: {decoder_params_fc}")
visualize_outputs(model_fc, test_loader)

# Convolutional Autoencoder
model_conv = ConvAutoencoder()
train_model(model_conv, train_loader)
encoder_params_conv, decoder_params_conv = count_encoder_decoder_parameters(model_conv)
print(f"Convolutional Autoencoder - Encoder Parameters: {encoder_params_conv}, Decoder Parameters: {decoder_params_conv}")
visualize_outputs(model_conv, test_loader)
