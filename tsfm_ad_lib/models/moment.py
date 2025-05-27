# tsfm_ad_lib/models/moment.py
from typing import Dict, Any, Optional
import torch

# Attempt to import MOMENTPipeline, provide guidance if not found
try:
    from momentfm import MOMENTPipeline
except ImportError:
    MOMENTPipeline = None
    print("Warning: momentfm library not found. Please install it to use MOMENT models.")
    print("You can typically install it using: pip install momentfm")

from .. import config # To access default model name

def load_moment_pipeline_for_reconstruction(
    pretrained_model_name_or_path: Optional[str] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
    device: Optional[str] = None,
    **kwargs: Any
) -> Optional['MOMENTPipeline']: # Return type is MOMENTPipeline or None if import failed
    """
    Loads a pre-trained MOMENT model pipeline, specifically configured for the
    'reconstruction' task, suitable for anomaly detection.

    Args:
        pretrained_model_name_or_path (Optional[str]): The name of the pre-trained model
            (e.g., "AutonLab/MOMENT-1-large") or path to local model files.
            If None, uses MOMENT_DEFAULT_MODEL_NAME from config.
        model_kwargs (Optional[Dict[str, Any]]): Additional keyword arguments for the model.
            The 'task_name' will be set to 'reconstruction' if not provided.
        device (Optional[str]): The device to load the model onto ('cuda', 'cpu').
                                If None, uses config.DEFAULT_DEVICE.
        **kwargs: Additional arguments passed to MOMENTPipeline.from_pretrained().

    Returns:
        Optional[MOMENTPipeline]: The loaded and initialized MOMENT pipeline, or None if
                                  the momentfm library could not be imported.
    
    Raises:
        ImportError: If the momentfm library is not installed and MOMENTPipeline is None.
        Exception: For errors during model loading from Hugging Face or local files.
    """
    if MOMENTPipeline is None:
        raise ImportError(
            "momentfm library is not installed. Cannot load MOMENT model. "
            "Please install it, e.g., 'pip install momentfm'"
        )

    if pretrained_model_name_or_path is None:
        pretrained_model_name_or_path = config.MOMENT_DEFAULT_MODEL_NAME

    # Ensure task_name is set for reconstruction
    current_model_kwargs = model_kwargs.copy() if model_kwargs else {}
    if 'task_name' not in current_model_kwargs:
        current_model_kwargs['task_name'] = 'reconstruction'
    elif current_model_kwargs['task_name'] != 'reconstruction':
        print(f"Warning: Overriding model_kwargs 'task_name' from '{current_model_kwargs['task_name']}' to 'reconstruction'.")
        current_model_kwargs['task_name'] = 'reconstruction'
    
    effective_device = device if device else config.DEFAULT_DEVICE

    print(f"Loading MOMENT model '{pretrained_model_name_or_path}' for task 'reconstruction' onto device '{effective_device}'...")
    
    try:
        pipeline = MOMENTPipeline.from_pretrained(
            pretrained_model_name_or_path,
            model_kwargs=current_model_kwargs,
            **kwargs
        )
        pipeline.init() # Initialize the model (as done in the original script)
        pipeline = pipeline.to(torch.device(effective_device))
        # Original script also cast to .float(). MOMENT models are usually float32.
        pipeline.float() 
        print("MOMENT model loaded and initialized successfully.")
        return pipeline
    except Exception as e:
        print(f"Error loading MOMENT model '{pretrained_model_name_or_path}': {e}")
        # Potentially re-raise or handle more gracefully
        raise


if __name__ == '__main__':
    # Example Usage (This will attempt to download the model if not cached)
    # Ensure you have an internet connection for the first run or have the model locally.
    if MOMENTPipeline is not None: # Only run if import was successful
        print("Attempting to load default MOMENT model for reconstruction...")
        try:
            # Test with default device from config (or 'cpu' if CUDA not available)
            test_device = config.DEFAULT_DEVICE if torch.cuda.is_available() else "cpu"
            
            moment_pipeline = load_moment_pipeline_for_reconstruction(device=test_device)
            
            if moment_pipeline:
                print(f"MOMENT pipeline loaded on device: {moment_pipeline.device}")
                print(f"Model class: {moment_pipeline.model.__class__.__name__}")

                # Example of how one might use it (conceptual)
                # dummy_input_moment = torch.randn(2, 1, config.MOMENT_DEFAULT_CONTEXT_LEN).to(moment_pipeline.device).float()
                # dummy_input_mask_moment = torch.ones_like(dummy_input_moment, dtype=torch.bool).to(moment_pipeline.device)
                # generated_random_mask = torch.ones(2, config.MOMENT_DEFAULT_CONTEXT_LEN, dtype=torch.bool).to(moment_pipeline.device) # Simplified
                
                # with torch.no_grad():
                #     output = moment_pipeline(x_enc=dummy_input_moment, 
                #                            input_mask=dummy_input_mask_moment, 
                #                            mask=generated_random_mask) # This 'mask' is for masked pretraining objective
                # print(f"Example output reconstruction shape: {output.reconstruction.shape}")

        except ImportError as e_imp:
            print(e_imp)
        except Exception as e_load:
            print(f"Could not run MOMENT example: {e_load}")
            print("This might be due to network issues, model availability, or momentfm setup.")
    else:
        print("Skipping MOMENT model loading example because momentfm library is not available.")