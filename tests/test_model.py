import torch
from src.models.autoencoder import Autoencoder

def test_model_init():
    """Test model initialization and basic forward pass."""
    model = Autoencoder()
    assert isinstance(model, torch.nn.Module)
    
    # Test shape
    x = torch.randn(1, 3, 256, 256)
    out = model(x)
    assert out.shape == (1, 3, 256, 256)

def test_model_load():
    """Test loading trained model weights from autoencoder.pth."""
    model = Autoencoder()
    state_dict = torch.load("models/bottle/autoencoder.pth", map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    
    # Check that model weights are loaded and eval mode is active
    assert not model.training
