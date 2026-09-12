import os
import sys
import csv
import argparse
from pathlib import Path
from collections import defaultdict
from PIL import Image
import numpy as np

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

def run_eda(
    dataset_root: str = "dataset/mvtec_anomaly_detection",
    categories: list = None,
    output_dir: str = "results/phase3"
):
    categories = categories or ['bottle', 'leather', 'transistor', 'zipper', 'screw']
    dataset_path = Path(dataset_root)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    print(f"========================================================")
    print(f"   MVTec AD Exploratory Data Analysis (EDA)")
    print(f"   Target Categories: {categories}")
    print(f"   Dataset Root:      {dataset_path.resolve()}")
    print(f"========================================================")
    
    summary_rows = []
    category_reports = {}
    
    for category in categories:
        cat_dir = dataset_path / category
        if not cat_dir.exists():
            print(f"Warning: Category directory {cat_dir} does not exist. Skipping.")
            continue
            
        print(f"\nAnalyzing category: {category.upper()}...")
        train_good_dir = cat_dir / "train" / "good"
        train_count = len(list(train_good_dir.glob("*.png"))) if train_good_dir.exists() else 0
        
        test_dir = cat_dir / "test"
        gt_dir = cat_dir / "ground_truth"
        
        test_breakdown = defaultdict(int)
        resolutions = set()
        mask_areas_px = []
        mask_areas_pct = []
        defect_mask_stats = defaultdict(lambda: {'count': 0, 'areas_px': [], 'areas_pct': []})
        
        if test_dir.exists():
            for defect_type_dir in sorted(test_dir.iterdir()):
                if not defect_type_dir.is_dir():
                    continue
                dtype = defect_type_dir.name
                imgs = sorted(defect_type_dir.glob("*.png"))
                test_breakdown[dtype] = len(imgs)
                
                for img_p in imgs:
                    with Image.open(img_p) as im:
                        resolutions.add(im.size) # (width, height)
                        w, h = im.size
                        total_px = w * h
                        
                    if dtype != 'good':
                        mask_p = gt_dir / dtype / f"{img_p.stem}_mask.png"
                        if mask_p.exists():
                            with Image.open(mask_p) as m_im:
                                m_arr = np.array(m_im.convert('L')) > 0
                                n_px = int(np.sum(m_arr))
                                pct = (n_px / total_px) * 100.0
                                mask_areas_px.append(n_px)
                                mask_areas_pct.append(pct)
                                defect_mask_stats[dtype]['count'] += 1
                                defect_mask_stats[dtype]['areas_px'].append(n_px)
                                defect_mask_stats[dtype]['areas_pct'].append(pct)
                                
        test_normal = test_breakdown.get('good', 0)
        test_defective = sum(count for dtype, count in test_breakdown.items() if dtype != 'good')
        total_test = test_normal + test_defective
        res_str = "; ".join([f"{w}x{h}" for w, h in sorted(resolutions)])
        
        min_px = min(mask_areas_px) if mask_areas_px else 0
        mean_px = np.mean(mask_areas_px) if mask_areas_px else 0.0
        max_px = max(mask_areas_px) if mask_areas_px else 0
        mean_pct = np.mean(mask_areas_pct) if mask_areas_pct else 0.0
        max_pct = max(mask_areas_pct) if mask_areas_pct else 0.0
        
        summary_rows.append({
            "category": category,
            "train_normal": train_count,
            "test_normal": test_normal,
            "test_defective": test_defective,
            "total_test": total_test,
            "num_defect_types": len([d for d in test_breakdown if d != 'good']),
            "resolution": res_str,
            "defect_area_mean_pct": f"{mean_pct:.2f}%",
            "defect_area_max_pct": f"{max_pct:.2f}%",
            "defect_area_mean_px": f"{mean_px:.0f}",
            "defect_types": ", ".join([f"{d} ({c})" for d, c in sorted(test_breakdown.items()) if d != 'good'])
        })
        
        category_reports[category] = {
            "train_count": train_count,
            "test_normal": test_normal,
            "test_defective": test_defective,
            "total_test": total_test,
            "resolutions": res_str,
            "test_breakdown": dict(test_breakdown),
            "mask_stats": {
                "min_px": min_px,
                "mean_px": mean_px,
                "max_px": max_px,
                "mean_pct": mean_pct,
                "max_pct": max_pct,
                "defect_breakdown": {
                    d: {
                        'count': s['count'],
                        'mean_px': float(np.mean(s['areas_px'])) if s['areas_px'] else 0.0,
                        'mean_pct': float(np.mean(s['areas_pct'])) if s['areas_pct'] else 0.0
                    } for d, s in defect_mask_stats.items()
                }
            }
        }
        
    # Save CSV
    csv_file = out_path / "eda_summary.csv"
    headers = [
        "category", "train_normal", "test_normal", "test_defective", "total_test",
        "num_defect_types", "resolution", "defect_area_mean_pct", "defect_area_max_pct",
        "defect_area_mean_px", "defect_types"
    ]
    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for r in summary_rows:
            writer.writerow(r)
    print(f"\nSaved CSV summary to: {csv_file.resolve()}")
    
    # Save Markdown Report
    md_file = out_path / "eda_report.md"
    with open(md_file, "w") as f:
        f.write("# MVTec Anomaly Detection — Phase 3.0 Dataset EDA Report\n\n")
        f.write("This report details the dataset distributions, image dimensions, defect breakdowns, ")
        f.write("and ground-truth mask pixel statistics across the 5 initial categories in VisionInspect Phase 3.0.\n\n")
        f.write("## 1. Summary Overview\n\n")
        f.write("| Category | Train Normal | Test Normal | Test Defective | Total Test | Defect Types | Resolution | Mean Defect Area | Max Defect Area |\n")
        f.write("|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n")
        for r in summary_rows:
            f.write(f"| **{r['category']}** | {r['train_normal']} | {r['test_normal']} | {r['test_defective']} | {r['total_test']} | {r['num_defect_types']} | {r['resolution']} | {r['defect_area_mean_pct']} | {r['defect_area_max_pct']} |\n")
        f.write("\n## 2. Category Detail Breakdown\n\n")
        for cat, data in category_reports.items():
            f.write(f"### Category: `{cat}`\n\n")
            f.write(f"- **Resolution**: {data['resolutions']}\n")
            f.write(f"- **Train Normal Images**: {data['train_count']}\n")
            f.write(f"- **Test Normal Images**: {data['test_normal']}\n")
            f.write(f"- **Test Defective Images**: {data['test_defective']}\n")
            f.write(f"- **Mean Defect Area**: {data['mask_stats']['mean_px']:.1f} px ({data['mask_stats']['mean_pct']:.2f}% of image)\n")
            f.write(f"- **Max Defect Area**: {data['mask_stats']['max_px']} px ({data['mask_stats']['max_pct']:.2f}% of image)\n\n")
            f.write("#### Defect Type Distribution\n\n")
            f.write("| Defect Type | Test Count | Mean Defect Area (px) | Mean Area (%) |\n")
            f.write("|---|:---:|:---:|:---:|\n")
            for dtype, count in sorted(data['test_breakdown'].items()):
                if dtype == 'good':
                    f.write(f"| `good` (normal) | {count} | - | - |\n")
                else:
                    d_stat = data['mask_stats']['defect_breakdown'].get(dtype, {})
                    m_px = d_stat.get('mean_px', 0.0)
                    m_pct = d_stat.get('mean_pct', 0.0)
                    f.write(f"| `{dtype}` | {count} | {m_px:.1f} px | {m_pct:.2f}% |\n")
            f.write("\n---\n\n")
            
        f.write("## 3. Engineering Insights for Multi-Category Detection\n\n")
        f.write("1. **Structural Objects vs. Textures**:\n")
        f.write("   - `leather` is a pure unaligned texture. Normal samples lack consistent spatial landmarks, making normal spatial priors counterproductive. `use_spatial_prior: false` enables uniform patch feature matching across texture surfaces.\n")
        f.write("   - `bottle`, `transistor`, `zipper`, and `screw` are structured objects with rigid geometry and background boundaries, benefiting significantly from spatial prior subtraction.\n")
        f.write("2. **Defect Scale Disparity**:\n")
        f.write("   - Defect areas vary widely: from small pinhole punctures and thread scratches (<0.5% image area) to large broken portions (>10% image area).\n")
        f.write("   - High-resolution feature extraction (64x64 grid, 4x4 px patch receptive field) is vital to preserve tiny defect signatures without spatial dilution.\n")
        f.write("3. **Category-Specific Distance Baselines**:\n")
        f.write("   - Feature distance norms differ fundamentally by product surface complexity. Thresholds must be calibrated per category rather than applied globally.\n")

    print(f"Saved Markdown report to: {md_file.resolve()}")
    return summary_rows

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MVTec AD Dataset Exploratory Data Analysis")
    parser.add_argument("--dataset-root", type=str, default="dataset/mvtec_anomaly_detection", help="Root directory of dataset")
    parser.add_argument("--output-dir", type=str, default="results/phase3", help="Output directory for reports")
    args = parser.parse_args()
    
    run_eda(dataset_root=args.dataset_root, output_dir=args.output_dir)
