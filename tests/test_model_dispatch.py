import subprocess
import pytest
from pathlib import Path
import sys

def test_model_dispatch_v22():
    v22_model = Path(__file__).resolve().parent.parent / 'models/bottle/patchcore_v22/memory_bank.pt'
    if not v22_model.exists():
        pytest.skip("Legacy patchcore_v22 memory bank not present")
    python_exe = sys.executable
    script_path = str(Path(__file__).resolve().parent.parent / 'inference.py')
    img_path = Path(__file__).resolve().parent.parent / 'dataset/mvtec_anomaly_detection/bottle/test/broken_large/000.png'
    if not img_path.exists():
        img_path = Path(__file__).resolve().parent.parent / 'assets/test_samples/bottle/broken_large_000.png'
    
    result = subprocess.run([python_exe, script_path, '--category', 'bottle', '--image', str(img_path), '--model', 'patchcore_v22'], capture_output=True, text=True)
    
    assert result.returncode == 0
    assert 'patchcore_v22' in result.stdout.lower()

def test_model_dispatch_v23():
    v23_model = Path(__file__).resolve().parent.parent / 'models/bottle/patchcore_v23/memory_bank.pt'
    if not v23_model.exists():
        pytest.skip("patchcore_v23 memory bank not present")
    python_exe = sys.executable
    script_path = str(Path(__file__).resolve().parent.parent / 'inference.py')
    img_path = Path(__file__).resolve().parent.parent / 'dataset/mvtec_anomaly_detection/bottle/test/broken_large/000.png'
    if not img_path.exists():
        img_path = Path(__file__).resolve().parent.parent / 'assets/test_samples/bottle/broken_large_000.png'
    
    result = subprocess.run([python_exe, script_path, '--category', 'bottle', '--image', str(img_path), '--model', 'patchcore_v23'], capture_output=True, text=True)
    
    assert result.returncode == 0
    assert 'patchcore_v23' in result.stdout.lower()
