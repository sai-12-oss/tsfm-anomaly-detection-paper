# tsfm_ad_lib/models/moment_utils.py
import torch

class Masking:
    """
    Generates a random mask for time series data, typically used during
    pre-training or fine-tuning of models like MOMENT.
    """
    def __init__(self, mask_ratio: float = 0.3):
        """
        Args:
            mask_ratio (float): The proportion of the time series to mask.
                                Should be between 0.0 and 1.0.
        """
        if not 0.0 <= mask_ratio <= 1.0:
            raise ValueError("mask_ratio must be between 0.0 and 1.0.")
        self.mask_ratio = mask_ratio
    
    def generate_mask(self, x: torch.Tensor, input_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Generates a boolean mask. True indicates a visible (unmasked) position,
        and False indicates a masked position.

        The original script's `input_mask` parameter in this method was not used in the
        mask generation logic itself. It was passed to the MOMENT model separately.
        This method focuses purely on generating the random mask based on `mask_ratio`.

        Args:
            x (torch.Tensor): The input tensor, expected to have shape 
                              [batch_size, num_channels, sequence_length] or
                              [batch_size, sequence_length] if num_channels is implicitly 1.
                              The mask is generated based on the sequence_length.
            input_mask (torch.Tensor, optional): An existing mask indicating valid data points.
                                                 While the original script passed this, it wasn't
                                                 used by this specific mask generation logic.
                                                 It's included for signature consistency if needed
                                                 for other parts of MOMENT's pipeline.

        Returns:
            torch.Tensor: A boolean tensor of shape [batch_size, sequence_length]
                          where False indicates a masked token.
        """
        if x.ndim == 3: # [batch_size, num_channels, sequence_length]
            batch_size, _, window_size = x.shape
        elif x.ndim == 2: # [batch_size, sequence_length]
            batch_size, window_size = x.shape
        else:
            raise ValueError(f"Input tensor x must have 2 or 3 dimensions, got {x.ndim}")

        if window_size == 0:
            return torch.empty((batch_size, 0), dtype=torch.bool, device=x.device)
            
        num_tokens_to_mask = int(window_size * self.mask_ratio)
        
        # Create a mask initialized to all True (visible)
        # Mask shape will be [batch_size, window_size]
        generated_random_mask = torch.ones((batch_size, window_size), dtype=torch.bool, device=x.device)
        
        for i in range(batch_size):
            if num_tokens_to_mask > 0:
                # Randomly select indices to mask (set to False)
                mask_indices = torch.randperm(window_size, device=x.device)[:num_tokens_to_mask]
                generated_random_mask[i, mask_indices] = False
                
        return generated_random_mask

if __name__ == '__main__':
    # Example Usage
    mask_ratio_test = 0.25
    masker = Masking(mask_ratio=mask_ratio_test)
    print(f"Masker initialized with mask_ratio: {masker.mask_ratio}")

    # Test with 2D input (batch_size, sequence_length)
    batch_s = 2
    seq_l = 10
    dummy_data_2d = torch.randn(batch_s, seq_l)
    print(f"\nTesting with 2D input shape: {dummy_data_2d.shape}")
    
    generated_m = masker.generate_mask(dummy_data_2d)
    print("Generated mask (2D input):")
    print(generated_m)
    if seq_l > 0:
        print(f"Number of False (masked) values in first sample: {(~generated_m[0]).sum().item()}")
        print(f"Expected masked values approx: {int(seq_l * mask_ratio_test)}")

    # Test with 3D input (batch_size, num_channels, sequence_length)
    num_ch = 1
    dummy_data_3d = torch.randn(batch_s, num_ch, seq_l)
    print(f"\nTesting with 3D input shape: {dummy_data_3d.shape}")
    
    generated_m_3d = masker.generate_mask(dummy_data_3d)
    print("Generated mask (3D input):")
    print(generated_m_3d)
    if seq_l > 0:
        print(f"Number of False (masked) values in first sample: {(~generated_m_3d[0]).sum().item()}")

    # Test with zero sequence length
    dummy_data_zero_len = torch.randn(batch_s, 0)
    print(f"\nTesting with zero sequence length input shape: {dummy_data_zero_len.shape}")
    generated_m_zero = masker.generate_mask(dummy_data_zero_len)
    print("Generated mask (zero sequence length input):")
    print(generated_m_zero)
    print(f"Shape of generated mask: {generated_m_zero.shape}")