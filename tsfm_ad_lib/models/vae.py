# tsfm_ad_lib/models/vae.py
import torch
import torch.nn as nn
from typing import Tuple

class VarEncoderDecoder(nn.Module):
    """
    Variational Autoencoder (VAE) model with a configurable number of hidden layers
    for encoding and decoding time series data.
    """
    def __init__(self, 
                 input_seq_length: int,
                 num_hidden_layers: int, 
                 initial_hidden_size: int, 
                 latent_dim: int = 32):
        """
        Args:
            input_seq_length (int): The length of the input sequence (e.g., 512).
            num_hidden_layers (int): Number of hidden layers in the encoder/decoder.
                                     The size of these layers will be halved successively.
            initial_hidden_size (int): Size of the first hidden layer after input.
                                       Subsequent layers will have their size halved.
            latent_dim (int): Dimensionality of the latent space (mu and log_var).
        """
        super().__init__()

        if not isinstance(input_seq_length, int) or input_seq_length <= 0:
            raise ValueError("input_seq_length must be a positive integer.")
        if not isinstance(num_hidden_layers, int) or num_hidden_layers < 0: # 0 hidden layers means direct to latent
            raise ValueError("num_hidden_layers must be a non-negative integer.")
        if not isinstance(initial_hidden_size, int) or initial_hidden_size <= 0:
            raise ValueError("initial_hidden_size must be a positive integer.")
        if not isinstance(latent_dim, int) or latent_dim <= 0:
            raise ValueError("latent_dim must be a positive integer.")

        self.input_seq_length = input_seq_length
        self.latent_dim = latent_dim

        # --- Encoder ---
        encoder_hidden_sizes = [input_seq_length]
        current_hidden_size = initial_hidden_size
        for _ in range(num_hidden_layers):
            encoder_hidden_sizes.append(current_hidden_size)
            if current_hidden_size // 2 <= 0: # Prevent hidden size from becoming 0 or negative
                # This might happen if initial_hidden_size is small and num_hidden_layers is large
                print(f"Warning: Encoder hidden size reduction resulted in size <= 0. Using 1 instead.")
                current_hidden_size = 1
            else:
                current_hidden_size //= 2
        
        # If num_hidden_layers is 0, encoder_hidden_sizes is just [input_seq_length]
        # The last size for fc_mu/fc_var should be the output of the last encoder layer
        # or initial_hidden_size if num_hidden_layers > 0, else input_seq_length
        
        final_encoder_layer_size = encoder_hidden_sizes[-1] if num_hidden_layers > 0 else initial_hidden_size
        if num_hidden_layers == 0 : # direct from input to latent_dim precursor
            # if no hidden layers, the "initial_hidden_size" acts as the pre-latent layer size
             encoder_hidden_sizes.append(initial_hidden_size) # this size will feed into mu/log_var
             final_encoder_layer_size = initial_hidden_size


        self.encoder_layers = nn.ModuleList()
        for i in range(len(encoder_hidden_sizes) -1): # Iterate up to the second to last size
            self.encoder_layers.append(nn.Linear(encoder_hidden_sizes[i], encoder_hidden_sizes[i+1]))
            self.encoder_layers.append(nn.BatchNorm1d(encoder_hidden_sizes[i+1]))
            self.encoder_layers.append(nn.ReLU())
        
        self.encoder = nn.Sequential(*self.encoder_layers)
        
        # Latent space layers
        self.fc_mu = nn.Linear(final_encoder_layer_size, latent_dim)
        self.fc_log_var = nn.Linear(final_encoder_layer_size, latent_dim)

        # --- Decoder ---
        # Decoder sizes will mirror encoder sizes in reverse, starting from the pre-latent size
        decoder_hidden_sizes = list(reversed(encoder_hidden_sizes)) # e.g. [final_enc_layer, ..., initial_hidden, input_seq]
        
        self.decoder_input_transform = nn.Linear(latent_dim, final_encoder_layer_size) # From latent to first decoder internal layer size
        
        self.decoder_layers = nn.ModuleList()
        # Iterate from final_encoder_layer_size up to input_seq_length
        for i in range(len(decoder_hidden_sizes) - 1):
            self.decoder_layers.append(nn.Linear(decoder_hidden_sizes[i], decoder_hidden_sizes[i+1]))
            self.decoder_layers.append(nn.BatchNorm1d(decoder_hidden_sizes[i+1]))
            # Original used LeakyReLU in decoder, ReLU in encoder. Last layer no activation.
            if i < len(decoder_hidden_sizes) - 2: # Apply activation to all but the output layer
                self.decoder_layers.append(nn.LeakyReLU()) 
            # If it's the last linear layer (outputting to input_seq_length), no activation or norm after it typically.
            # However, the original code applied norm+activation even to the last layer. Let's refine.
            # The original code pattern was [Linear, Norm, Activation] for all.
            # For the final output layer, usually no activation or a specific one (e.g. Sigmoid if inputs are 0-1)
            # Here, inputs are scaled meter readings, so linear output is fine.
            # The original structure might have BatchNorm on the output layer, which is unusual.
            # Let's keep it for now to match, but flag for review.
            # If this is the last layer leading to final output, consider if BatchNorm1d is appropriate.
            # It's typically not used on the very last output layer.
            # For replication, the original pattern was: Linear, Norm, Activation.
            # The last layer in the original decoder was Linear(..., seq_length), Norm, Activation.

        self.decoder = nn.Sequential(*self.decoder_layers)
        
    def _reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick to sample from N(mu, var) by sampling from N(0,1).
        Args:
            mu (torch.Tensor): Mean of the latent Gaussian.
            log_var (torch.Tensor): Log variance of the latent Gaussian.
        Returns:
            torch.Tensor: Sampled latent vector.
        """
        std = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(std)
        return mu + epsilon * std

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the VAE.
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, input_seq_length].
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 
                - reconstructed_x (torch.Tensor): Reconstructed input tensor.
                - mu (torch.Tensor): Latent mean.
                - log_var (torch.Tensor): Latent log variance.
        """
        # Encoder
        h = self.encoder(x)
        mu = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        
        # Reparameterization
        z = self._reparameterize(mu, log_var)
        
        # Decoder
        reconstructed_x_intermediate = self.decoder_input_transform(z)
        reconstructed_x = self.decoder(reconstructed_x_intermediate)
        
        return reconstructed_x, mu, log_var

    def loss_function(self, reconstructed_x: torch.Tensor, x: torch.Tensor, 
                      mu: torch.Tensor, log_var: torch.Tensor, 
                      reconstruction_loss_fn = nn.MSELoss(reduction='sum'), 
                      kld_weight: float = 1.0) -> torch.Tensor:
        """
        Calculates the VAE loss (reconstruction + KLD).
        Args:
            reconstructed_x: Reconstructed input data.
            x: Original input data.
            mu: Latent mean.
            log_var: Latent log variance.
            reconstruction_loss_fn: The loss function for reconstruction (e.g., MSE). Default sum.
            kld_weight (float): Weight for the KL divergence term.
        Returns:
            torch.Tensor: The total VAE loss.
        """
        reconstruction_loss = reconstruction_loss_fn(reconstructed_x, x)
        
        # KL divergence: 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        
        total_loss = reconstruction_loss + kld_weight * kld
        return total_loss / x.size(0) # Average loss per batch element


if __name__ == '__main__':
    # Example Usage
    batch_size = 16
    seq_len = 512 # from VAE_DEFAULT_INPUT_SIZE in config
    num_layers = 1 # from original script's Optuna search for VAE
    hidden_sz = 256 # example from original script
    lat_dim = 32 # example

    # Create model instance
    model = VarEncoderDecoder(
        input_seq_length=seq_len,
        num_hidden_layers=num_layers,
        initial_hidden_size=hidden_sz,
        latent_dim=lat_dim
    )
    print("VAE Model Architecture:")
    print(model)

    # Create a dummy input tensor (ensure it's float for nn.Linear)
    dummy_input = torch.randn(batch_size, seq_len).float() 
    # VAE in original script used .double(), but .float() is more common for PyTorch.
    # If .double() is strictly needed, model parameters also need to be .double().

    # Perform a forward pass
    try:
        reconstructed, mu, log_var = model(dummy_input)
        print(f"\nInput shape: {dummy_input.shape}")
        print(f"Reconstructed shape: {reconstructed.shape}")
        print(f"Mu shape: {mu.shape}")
        print(f"Log_var shape: {log_var.shape}")

        # Test loss function
        loss = model.loss_function(reconstructed, dummy_input, mu, log_var)
        print(f"Example VAE Loss: {loss.item()}")

    except Exception as e:
        print(f"Error during VAE forward pass example: {e}")
        import traceback
        traceback.print_exc()

    # Test case: 0 hidden layers
    print("\n--- VAE Model with 0 hidden layers ---")
    model_0_layers = VarEncoderDecoder(
        input_seq_length=seq_len,
        num_hidden_layers=0, # No hidden layers
        initial_hidden_size=128, # This will be the size before latent space
        latent_dim=lat_dim
    )
    print(model_0_layers)
    try:
        reconstructed, mu, log_var = model_0_layers(dummy_input)
        print(f"Reconstructed shape (0 layers): {reconstructed.shape}")
        loss_0_layers = model_0_layers.loss_function(reconstructed, dummy_input, mu, log_var)
        print(f"Example VAE Loss (0 layers): {loss_0_layers.item()}")
    except Exception as e:
        print(f"Error during VAE (0 layers) forward pass example: {e}")