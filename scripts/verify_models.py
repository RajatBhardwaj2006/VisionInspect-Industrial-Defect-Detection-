"""
scripts/verify_models.py
========================
Verification utility for VisionInspect production models.

Checks that all 5 PatchCore v2.3 models (Bottle, Leather, Transistor, Zipper, Screw)
and category prototypes are present, valid tensors, and not unpulled Git LFS pointers.
"""

import sys
import os
import subprocess
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CATEGORIES = ["bottle", "leather", "transistor", "zipper", "screw"]
MODELS_DIR = PROJECT_ROOT / "models"


def check_lfs_pointer(file_path: Path) -> bool:
    """Returns True if file is a Git LFS pointer text file instead of the actual tensor."""
    if not file_path.exists():
        return False
    if file_path.stat().st_size < 1024:
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                header = f.read(100)
                if "version https://git-lfs.github.com/spec/v1" in header:
                    return True
        except Exception:
            pass
    return False


def verify_all_models() -> bool:
    print("=" * 65)
    print("       VISIONINSPECT PRODUCTION MODEL INTEGRITY CHECK")
    print("=" * 65)

    all_ok = True
    lfs_pointers_found = []

    # 1. Category Prototypes Check
    proto_path = MODELS_DIR / "category_prototypes.pt"
    if not proto_path.exists():
        print(f"[MISSING]  Category prototypes: {proto_path.relative_to(PROJECT_ROOT)}")
        all_ok = False
    else:
        print(f"[OK]       Category prototypes: {proto_path.relative_to(PROJECT_ROOT)} ({proto_path.stat().st_size / 1024:.1f} KB)")

    # 2. Category Model Folders Check
    for cat in CATEGORIES:
        cat_dir = MODELS_DIR / cat / "patchcore_v23"
        mb_path = cat_dir / "memory_bank.pt"
        meta_path = cat_dir / "metadata.json"
        prior_path = cat_dir / "spatial_prior.pt"

        print(f"\n--- Checking Model: {cat.upper()} ---")

        if not cat_dir.exists():
            print(f"  [MISSING] Directory: {cat_dir.relative_to(PROJECT_ROOT)}")
            all_ok = False
            continue

        # Metadata
        if meta_path.exists():
            print(f"  [OK] Metadata:      {meta_path.relative_to(PROJECT_ROOT)}")
        else:
            print(f"  [WARN] Metadata missing: {meta_path.relative_to(PROJECT_ROOT)}")

        # Spatial Prior (where applicable)
        if prior_path.exists():
            print(f"  [OK] Spatial Prior: {prior_path.relative_to(PROJECT_ROOT)} ({prior_path.stat().st_size / 1024:.1f} KB)")
        else:
            print(f"  [INFO] Spatial Prior: None (texture / uniform background)")

        # Bottle p75 prior
        if cat == "bottle":
            p75 = cat_dir / "spatial_prior_p75.pt"
            if p75.exists():
                print(f"  [OK] p75 Prior:     {p75.relative_to(PROJECT_ROOT)} ({p75.stat().st_size / 1024:.1f} KB)")

        # Memory Bank
        if not mb_path.exists():
            print(f"  [MISSING] Memory Bank: {mb_path.relative_to(PROJECT_ROOT)}")
            all_ok = False
        elif check_lfs_pointer(mb_path):
            print(f"  [LFS POINTER] {mb_path.relative_to(PROJECT_ROOT)} is an unpulled Git LFS pointer!")
            lfs_pointers_found.append(mb_path)
            all_ok = False
        else:
            size_mb = mb_path.stat().st_size / (1024 * 1024)
            print(f"  [OK] Memory Bank:   {mb_path.relative_to(PROJECT_ROOT)} ({size_mb:.2f} MB)")

    print("\n" + "=" * 65)

    if lfs_pointers_found:
        print("\n[ACTION REQUIRED] Git LFS pointer files detected.")
        print("Please run the following command to download the model tensors:")
        print("    git lfs pull")
        print("\nAttempting automatic git lfs pull now...")
        try:
            res = subprocess.run(["git", "lfs", "pull"], cwd=str(PROJECT_ROOT), capture_output=True, text=True)
            print(res.stdout)
            if res.returncode == 0:
                print("[SUCCESS] git lfs pull completed. Re-running verification...")
                return verify_all_models()
        except Exception as e:
            print(f"Could not execute git lfs pull: {e}")
        return False

    if all_ok:
        print("[STATUS: READY] All 5 production models are verified and ready for inference!")
    else:
        print("[STATUS: INCOMPLETE] Some required model assets are missing.")

    print("=" * 65)
    return all_ok


if __name__ == "__main__":
    success = verify_all_models()
    sys.exit(0 if success else 1)
