import torch
import torch.nn as nn

# Define the PseudoLoss class which calculates the loss used in the DDL model
class PseudoLoss(nn.Module):
    def __init__(self, temporal=1):
        super().__init__()
        # Define the target frame for spatiotemporal data
        self.target_frame = (temporal - 1) // 2
        
    def non_zero_mean(self, x):
        # Calculate the mean of non-zero elements in a tensor
        non_zero_tuple = torch.nonzero(x, as_tuple=True)
        return x[non_zero_tuple].mean() if len(non_zero_tuple[0]) else torch.zeros(1).to(x.device)

    def forward(self, x, x_anom, normal_reconstruction, anomalous_reconstruction, mask, anomaly_weight, lambda_weight=1e-2):
        # Select the target frame if the input has a temporal dimension
        if len(x.shape) == 5:
            x = x[:, :, self.target_frame]
        if len(x_anom.shape) == 5:
            x_anom = x_anom[:, :, self.target_frame]
        if len(mask.shape) == 5:
            mask = mask[:, :, self.target_frame]
        mask = mask[:, 0]
                    
        # Initialize the loss dictionary
        loss = {'loss': torch.zeros(1).to(x.device)}
        dim = x.shape[1]

        # Calculate the reconstruction loss
        recon_loss = torch.norm(x - normal_reconstruction, dim=1).mean() / (dim ** 0.5)
        loss['recon_loss'] = recon_loss

        # Calculate the positive and negative loss for the distinction loss
        pos_loss = mask * torch.norm(x - anomalous_reconstruction, dim=1) / (dim ** 0.5)
        neg_loss = mask * torch.norm(x_anom - anomalous_reconstruction, dim=1) / (dim ** 0.5)
        
        error = 1e-6
        masked_error = mask * error
        
        # Calculate the distinction loss
        dist_loss = self.non_zero_mean((pos_loss + masked_error) / (neg_loss + error))
        loss['dist_loss'] = lambda_weight * dist_loss

        # Include the anomaly weight in the loss dictionary
        loss['anomaly_weight'] = anomaly_weight
        
        # Combine the losses into the total loss
        loss['loss'] += recon_loss + lambda_weight * dist_loss

        return loss

# Define the PseudoAnomalyCreator class which generates pseudo anomalies for training
class PseudoAnomalyCreator(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        # Initialize the anomaly weight parameter
        self.anomaly_weight = nn.Parameter(5 * torch.zeros(1))

    def forward(self, x, mask):
        # Calculate the anomaly weight using a sigmoid function
        anomaly_weight = self.sigmoid(self.anomaly_weight).to(x.device)
        # Generate random noise to create pseudo anomalies
        noise = torch.rand_like(x).to(x.device)
        anomalous_x = (1 - anomaly_weight) * x + anomaly_weight * noise
        anomalous_x_object = (1 - mask) * x + mask * anomalous_x
        return anomalous_x_object, anomaly_weight
