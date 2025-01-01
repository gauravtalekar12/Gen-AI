import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
# from tqdm import tqdm
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
torch.cuda.empty_cache()



transform = transforms.Compose([
    transforms.Resize((128, 128)), 
    transforms.ToTensor(),
    #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
train_dataset = ImageFolder(root="./afhq/train", transform=transform)
val_dataset = ImageFolder(root="./afhq/val", transform=transform)
train_dataloader = DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True,
    num_workers=4
)
val_dataloader = DataLoader(
    val_dataset,
    batch_size=128,
    shuffle=True,
    num_workers=4
)
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")






import torch
import torch.nn as nn
device = torch.device("cuda:2")


class Generator(nn.Module):
    def __init__(self, latent_dim, image_channels):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.BatchNorm1d(2048),
            nn.Linear(2048, 4096),
            nn.ReLU(),
            nn.BatchNorm1d(4096),
            nn.Linear(4096, 8192),
            nn.ReLU(),
            nn.BatchNorm1d(8192),
            nn.Linear(8192, 128 * 128 * image_channels),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z).view(z.shape[0], -1, 128, 128)

# Define the forward pass discriminator with batch normalization and dropout
class ForwardDiscriminator(nn.Module):
    def __init__(self, image_channels):
        super(ForwardDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(128 * 128 * image_channels, 2048),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(2048),  # Add batch normalization
            nn.Linear(2048, 1024),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(1024),  # Add batch normalization
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(512),  # Add batch normalization
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),  # Add batch normalization
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x.view(x.shape[0], -1))


# Define the backward pass discriminator with batch normalization and dropout
class BackwardDiscriminator(nn.Module):
    def __init__(self, latent_dim):
        super(BackwardDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),  # Add batch normalization
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(512),  # Add batch normalization
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(1024),  # Add batch normalization
            nn.Linear(1024, 2048),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(2048),  # Add batch normalization
            nn.Linear(2048, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.model(x.view(x.shape[0], -1))

# Define the encoder with batch normalization and dropout
class Encoder(nn.Module):
    def __init__(self, image_channels, latent_dim):
        super(Encoder, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(128 * 128 * image_channels, 1024),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(1024),  # Add batch normalization
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(512),  # Add batch normalization
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),  # Add batch normalization
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(128),  # Add batch normalization
            nn.Linear(128, latent_dim)
        )
    
    def forward(self, x):
        return self.model(x.view(x.shape[0], -1))

# Initialize the models
latent_dim = 100
image_channels = 3  # Adjust based on your dataset

generator = Generator(latent_dim, image_channels)
forward_discriminator = ForwardDiscriminator(image_channels)
backward_discriminator = BackwardDiscriminator(latent_dim)
encoder = Encoder(image_channels, latent_dim)







import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

# Define the Bi-GAN components and optimizers
device = torch.device("cuda:2")
latent_dim = 100
image_channels = 3
generator = Generator(latent_dim, image_channels)
forward_discriminator = ForwardDiscriminator(image_channels)
backward_discriminator = BackwardDiscriminator(latent_dim)
encoder = Encoder(image_channels, latent_dim)


# Move your models and tensors to the CPU
generator = generator.to(device)
forward_discriminator = forward_discriminator.to(device)
backward_discriminator = backward_discriminator.to(device)
encoder = encoder.to(device)

# Define loss functions
adversarial_loss = nn.BCELoss()
pixelwise_loss = nn.MSELoss()
encoder_loss = nn.MSELoss()  # You can choose the appropriate loss for encoder

# Optimizers
optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_F = optim.Adam(forward_discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_B = optim.Adam(backward_discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_E = optim.Adam(encoder.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Training parameters
num_epochs = 100
batch_size = 32

# Training loop
for epoch in range(num_epochs):
    for i, batch in enumerate(train_dataloader):
        real_images, _ = batch
        real_images = real_images.to(device)

        # Train the forward discriminator
        optimizer_F.zero_grad()

        # Sample noise for the generator
        z = Variable(torch.randn(real_images.size(0), latent_dim))
        z = z.to(device)

        # Generate fake images
        fake_images = generator(z)

        # Compute forward discriminator loss for real and fake images
        real_labels = torch.ones(real_images.size(0), 1).to(device)  # Use torch.ones with the batch size
        fake_labels = torch.zeros(fake_images.size(0), 1).to(device)  # Use torch.zeros with the batch size

        real_loss = adversarial_loss(forward_discriminator(real_images), real_labels)
        fake_loss = adversarial_loss(forward_discriminator(fake_images.detach()), fake_labels)
        forward_discriminator_loss = real_loss + fake_loss

        forward_discriminator_loss.backward()
        optimizer_F.step()

        # Train the backward discriminator
        optimizer_B.zero_grad()

        # Encode real images
        encoded_latents = encoder(real_images)

        # Sample noise for the backward discriminator
        z = Variable(torch.randn(encoded_latents.size(0), latent_dim))
        z = z.to(device)

        # Compute backward discriminator loss for real and generated latents
        real_labels = real_labels.view(-1, 1).float()
        fake_labels = fake_labels.view(-1, 1).float()

        real_loss = adversarial_loss(backward_discriminator(encoded_latents), real_labels)
        fake_loss = adversarial_loss(backward_discriminator(z), fake_labels)
        backward_discriminator_loss = real_loss + fake_loss

        backward_discriminator_loss.backward()
        optimizer_B.step()

        # Train the generator
        optimizer_G.zero_grad()

        # Generate fake images again
        z = Variable(torch.randn(real_images.size(0), latent_dim))
        z = z.to(device)
        fake_images = generator(z)

        # Compute generator loss
        generator_loss = adversarial_loss(forward_discriminator(fake_images), real_labels)
        pixel_loss = pixelwise_loss(fake_images, real_images)

        total_generator_loss = generator_loss + pixel_loss

        total_generator_loss.backward()
        optimizer_G.step()

        # Train the encoder
        optimizer_E.zero_grad()

        # Encode real images and sample noise
        encoded_latents = encoder(real_images)
        z = Variable(torch.randn(encoded_latents.size(0), latent_dim))
        z = z.to(device)

        # Compute encoder loss
        encoder_loss_value = encoder_loss(encoded_latents, z)

        encoder_loss_value.backward()
        optimizer_E.step()

        # Print progress
    print(f"[Epoch {epoch}/{num_epochs}] [Batch {i}/{len(train_dataloader)}] "
              f"[Forward D loss: {forward_discriminator_loss.item():.4f}] "
              f"[Backward D loss: {backward_discriminator_loss.item():.4f}] "
              f"[G loss: {generator_loss.item():.4f}] "
              f"[Encoder loss: {encoder_loss_value.item():.4f}]")
    
    # Save generated images at the end of each epoch
    if epoch % 10 == 0:
        fake_images = generator(z)
        # Save or visualize the generated images

# Save the trained models if needed
torch.save(generator.state_dict(), 'generator.pth')
torch.save(forward_discriminator.state_dict(), 'forward_discriminator.pth')
torch.save(backward_discriminator.state_dict(), 'backward_discriminator.pth')
torch.save(encoder.state_dict(), 'encoder.pth')





import matplotlib.pyplot as plt
import numpy as np

# Set the generator to evaluation mode
generator.eval()

# Number of rows and columns in the grid
n_rows, n_cols = 10, 10

# Generate random noise vectors
z = torch.randn(n_rows * n_cols, latent_dim).to(device)

# Generate fake images
with torch.no_grad():
    fake_images = generator(z)

# Reshape the fake_images tensor for plotting
fake_images = fake_images.view(-1, 3, 128, 128)  # Assuming 3 channels and 128x128 resolution

# Create a grid of generated images
grid = np.zeros((n_rows * 128, n_cols * 128, 3), dtype=np.uint8)

for i in range(n_rows):
    for j in range(n_cols):
        image = fake_images[i * n_cols + j].cpu().numpy().transpose((1, 2, 0))
        image = ((image + 1) / 2) * 255  # Scale to [0, 255]
        image = np.uint8(image)
        grid[i * 128: (i + 1) * 128, j * 128: (j + 1) * 128, :] = image

# Plot the grid of generated images
plt.figure(figsize=(10, 10))
plt.imshow(grid)
plt.axis('off')
plt.show()
