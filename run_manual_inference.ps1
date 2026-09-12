$images_v23 = @('broken_large/000.png', 'good/001.png', 'broken_small/000.png', 'contamination/000.png')
foreach ($img in $images_v23) {
    Write-Output "--- Running V2.3 on $img ---"
    & "X:\VScode\Artificial_intelligence_n_Machine_learning\VisionInspect\visioninspect_py312\Scripts\python.exe" inference.py --category bottle --image "dataset/mvtec_anomaly_detection/bottle/test/$img" --model patchcore_v23
}
Write-Output "--- Running V2.2 on broken_large/000.png ---"
& "X:\VScode\Artificial_intelligence_n_Machine_learning\VisionInspect\visioninspect_py312\Scripts\python.exe" inference.py --category bottle --image "dataset/mvtec_anomaly_detection/bottle/test/broken_large/000.png" --model patchcore_v22
