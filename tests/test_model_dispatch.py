import subprocess
import pytest
from pathlib import Path
import sys

def test_model_dispatch_v22():
    python_exe = sys.executable
    script_path = str(Path(__file__).resolve().parent.parent / 'inference.py')
    img_path = str(Path(__file__).resolve().parent.parent / 'dataset/mvtec_anomaly_detection/bottle/test/broken_large/000.png')
    
    result = subprocess.run([python_exe, script_path, '--category', 'bottle', '--image', img_path, '--model', 'patchcore_v22'], capture_output=True, text=True)
    
    assert result.returncode == 0
    assert 'PatchCore V2.2 model loaded from: models\bottle\patchcore_v22' in result.stdout or 'PatchCore V2.2 model loaded from: models/bottle/patchcore_v22' in result.stdout or 'patchcore_v22' in result.stdout

def test_model_dispatch_v23():
    python_exe = sys.executable
    script_path = str(Path(__file__).resolve().parent.parent / 'inference.py')
    img_path = str(Path(__file__).resolve().parent.parent / 'dataset/mvtec_anomaly_detection/bottle/test/broken_large/000.png')
    
    result = subprocess.run([python_exe, script_path, '--category', 'bottle', '--image', img_path, '--model', 'patchcore_v23'], capture_output=True, text=True)
    
    assert result.returncode == 0
    assert 'PatchCore V2.3 model loaded from: models\bottle\patchcore_v23' in result.stdout or 'PatchCore V2.3 model loaded from: models/bottle/patchcore_v23' in result.stdout or 'patchcore_v23' in result.stdout
