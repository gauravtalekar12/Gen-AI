Introduction
This repository contains the implementation and results of experiments conducted using three popular types of generative models: Variational Autoencoders (VAE), Generative Adversarial Networks (GAN), and Denoising Diffusion Probabilistic Models (DDPM). These models have been utilized to generate synthetic images based on a dataset of animal images.

Experiments
Variational Autoencoders (VAE)
Model Description: The VAE model consists of a convolutional neural network (CNN) encoder and decoder. The latent dimension is set to 64 for image reconstruction and generation.
Training Details: Trained using Adam optimizer with a learning rate of 1e-3 over 10 epochs.
Results: The VAE was able to generate new samples that display transitions between different animal images, suggesting a continuous latent space. The interpolated images demonstrate smooth transitions, indicating that the VAE captures meaningful features of the data.
Generative Adversarial Networks (GAN)
Model Description: Implemented a Deep Convolutional GAN (DCGAN) with a discriminator and generator architecture using CNNs. The discriminator consists of 5 convolutional layers, while the generator uses transposed convolutions to upsample the latent noise vector to image dimensions.
Training Dynamics: Multiple experiments were conducted altering the ratio of generator to discriminator training steps to explore the balance between the two networks and its effect on the training stability and image quality.
Results: The generated images vary in quality with different training setups. Training the discriminator less frequently compared to the generator helped in maintaining a balance and avoiding mode collapse, a common issue in GAN training.
Denoising Diffusion Probabilistic Models (DDPM)
Model Description: DDPMs were implemented to generate high-fidelity images through a process of gradually denoising a signal. The model uses a scheduler to adjust the variance of the noise during training.
Training Details: The models were trained using different numbers of timesteps to explore the impact on image quality.
Results: Using more timesteps generally led to better generation quality, highlighting the trade-off between computational cost and fidelity.
Key Observations
VAE: Provides a good balance between performance and computational efficiency. The ability to encode and decode images efficiently makes VAEs suitable for tasks where model interpretability and intermediate representations are beneficial.
GAN: While capable of generating high-quality images, GANs require careful tuning of training dynamics to avoid common pitfalls like mode collapse and discriminator overpowering.
DDPM: Exhibits excellent potential in generating detailed images but at a higher computational and time cost. Optimal performance is achieved with a higher number of timesteps.
