
import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from srgan import Enlarger_PixelShuffle, Discriminator_Naive  # Make sure these are PyTorch versions
import vgg  # Your VGG model adapted for PyTorch
import torch.nn as nn
from torch.optim import lr_scheduler, SGD
import cv2

# Custom dataset
class TrainData(Dataset):
    def __init__(self, hr_img_path, hr_trans, lr_trans):
        # Assuming hr_img_path is a directory containing HR images
        self.hr_img_filenames = [os.path.join(hr_img_path, f) for f in os.listdir(hr_img_path) if os.path.isfile(os.path.join(hr_img_path, f))]
        self.hr_trans = hr_trans
        self.lr_trans = lr_trans

    def __getitem__(self, index):
        # Load the HR image
        hr_img = cv2.imread(self.hr_img_filenames[index])
        hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB format

        # Apply transformations
        hr_img = self.hr_trans(hr_img)
        lr_img = self.lr_trans(hr_img)

        return lr_img, hr_img

    def __len__(self):
        return len(self.hr_img_filenames)

def train(G, D, VGG, train_dataset, n_epoch_init, n_epoch, batch_size, checkpoint_dir, device):

    G.train()
    D.train()
    VGG.eval()

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    n_step_epoch = len(train_loader)

    # Define the learning rate schedule and optimizer
    lr = 0.05
    g_optimizer_init = SGD(G.parameters(), lr=lr, momentum=0.9)
    g_optimizer = SGD(G.parameters(), lr=lr, momentum=0.9)
    d_optimizer = SGD(D.parameters(), lr=lr, momentum=0.9)

    # Learning rate scheduler
    lr_scheduler_G = lr_scheduler.StepLR(g_optimizer, step_size=1000, gamma=0.1)
    lr_scheduler_D = lr_scheduler.StepLR(d_optimizer, step_size=1000, gamma=0.1)

    # Loss functions
    criterion_GAN = nn.BCEWithLogitsLoss()
    criterion_content = nn.MSELoss()

    n_step_epoch = len(train_dataset) / batch_size

    # Training loop for generator initialization
    for epoch in range(0):
        for step, (lr_patches, hr_patches) in enumerate(train_loader):
            lr_patches, hr_patches = lr_patches.to(device), hr_patches.to(device)

            G.zero_grad()
            gen_hr = G(lr_patches)
            g_loss_init = criterion_content(gen_hr, hr_patches)
            g_loss_init.backward()
            g_optimizer_init.step()

            print(f"Epoch: [{epoch+1}/{n_epoch_init}] step: [{step+1}/{n_step_epoch}] mse: {g_loss_init.item()}")

    # Adversarial training
    for epoch in range(n_epoch):
        for step, (lr_patches, hr_patches) in enumerate(train_loader):
            lr_patches, hr_patches = lr_patches.to(device), hr_patches.to(device)

            # Train discriminator
            D.zero_grad()
            real_out = D(hr_patches).view(-1)
            fake_out = D(G(lr_patches).detach()).view(-1)
            d_loss_real = criterion_GAN(real_out, torch.ones_like(real_out))
            d_loss_fake = criterion_GAN(fake_out, torch.zeros_like(fake_out))
            d_loss = (d_loss_real + d_loss_fake) / 2
            d_loss.backward()
            d_optimizer.step()

            # Train generator
            G.zero_grad()
            fake_hr = G(lr_patches)
            fake_out = D(fake_hr).view(-1)
            g_loss_gan = criterion_GAN(fake_out, torch.ones_like(fake_out))
            g_loss_content = criterion_content(fake_hr, hr_patches)

            # Feature loss
            gen_features = VGG(fake_hr)
            real_features = VGG(hr_patches)
            g_loss_feature = criterion_content(gen_features, real_features)

            g_loss = g_loss_gan + g_loss_content + 1e-3 * g_loss_feature
            g_loss.backward()
            g_optimizer.step()

            print(f"Epoch: [{epoch+1}/{n_epoch}] step: [{step+1}/{n_step_epoch}] g_loss: {g_loss.item()}, d_loss: {d_loss.item()}")

            # Update learning rates
            lr_scheduler_G.step()
            lr_scheduler_D.step()

        if (epoch != 0) and (epoch % 10 == 0):
            torch.save(G.state_dict(), os.path.join(checkpoint_dir, f'g_{epoch}.pth'))
            torch.save(D.state_dict(), os.path.join(checkpoint_dir, f'd_{epoch}.pth'))

if __name__ == '__main__':

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ###====================== HYPER-PARAMETERS ===========================###
    batch_size = 8
    n_epoch_init = 100  # You should define these in your config or here directly
    n_epoch = 2000
    # create folders to save result images and trained models
    save_dir = "samples"
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_dir = "models"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Transformations for HR and LR images
    hr_transform = transforms.Compose([
        transforms.ToPILImage(),  # Convert numpy arrays to PIL images
        transforms.RandomCrop(size=(384, 384)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    lr_transform = transforms.Compose([
        transforms.ToPILImage(),  # Convert numpy arrays to PIL images
        transforms.Resize(size=(96, 96)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    G = Enlarger_PixelShuffle().cuda()
    D = Discriminator_Naive().cuda()
    # Usage
    VGG = vgg.VGG19FeatureExtractor()
    train_dataset = TrainData("DIV2K/DIV2K_train_HR/", hr_transform, lr_transform)



    train(G, D, VGG, train_dataset, n_epoch_init, n_epoch, batch_size, checkpoint_dir, device)