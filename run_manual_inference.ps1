# VisionInspect - Manual CLI Inference Verification Script
$PythonExe = if (Test-Path "visioninspect_py312\Scripts\python.exe") { "visioninspect_py312\Scripts\python.exe" } elseif (Test-Path ".venv\Scripts\python.exe") { ".venv\Scripts\python.exe" } else { "python" }

$images_v23 = @('broken_large/000.png', 'good/001.png', 'broken_small/000.png', 'contamination/000.png')
foreach ($img in $images_v23) {
    Write-Output "--- Running V2.3 on $img ---"
    & $PythonExe inference.py --category bottle --image "dataset/mvtec_anomaly_detection/bottle/test/$img" --model patchcore_v23
}
Write-Output "--- Running V2.2 on broken_large/000.png ---"
& $PythonExe inference.py --category bottle --image "dataset/mvtec_anomaly_detection/bottle/test/broken_large/000.png" --model patchcore_v22
