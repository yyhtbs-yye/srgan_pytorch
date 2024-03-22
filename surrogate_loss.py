import torch
import torch.nn as nn
import pytorch_wavelets as ptwt

class WaveletMSELoss(nn.Module):
    def __init__(self, J=2, wave='db1'):
        super(WaveletMSELoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.dwt = ptwt.DWTForward(J=J, wave=wave, mode='zero')

        # Initialize a learnable weight for each resolution level including the approximation
        # Level 0 corresponds to the approximation coefficients
        self.weights = nn.Parameter(torch.ones(J + 1), requires_grad=True)

    def forward(self, hat_y, y):
        # Compute the standard MSE
        pixel_mse = self.mse_loss(hat_y, y)

        # Compute wavelet coefficients for both images
        Yl_hat, Yh_hat = self.dwt(hat_y)
        Yl, Yh = self.dwt(y)

        # Start wavelet MSE with the weighted loss of the approximation coefficients
        wavelet_mse = self.weights[0] * self.mse_loss(Yl_hat, Yl)

        # Add the weighted loss from the detail coefficients at each level
        for i in range(len(Yh_hat)):
            wavelet_mse += self.weights[i + 1] * sum(
                self.mse_loss(Yh_hat[i][j], Yh[i][j]) for j in range(len(Yh_hat[i]))
            )

        # Combine the pixel-wise MSE with the weighted wavelet MSE
        total_loss = pixel_mse + wavelet_mse
        return total_loss
