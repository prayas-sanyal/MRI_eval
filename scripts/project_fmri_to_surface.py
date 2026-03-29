#Project volumetric fMRI (NIfTI) onto the fsaverage5 cortical surface.

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
from nilearn import datasets, surface


def project_volume_to_surface(
    nifti_path: str | Path,
    mesh: str = "fsaverage5",
    radius: float = 3.0,
) -> np.ndarray:
    """project a 3D/4D NIfTI onto both fsaverage hemispheres.

    returns (n_vertices_total, n_timepoints).
    """
    fsaverage = datasets.fetch_surf_fsaverage(mesh)
    img = nib.load(str(nifti_path))

    hemis = []
    for hemi in ("left", "right"):
        surf_data = surface.vol_to_surf(
            img,
            surf_mesh=fsaverage[f"pial_{hemi}"],
            inner_mesh=fsaverage[f"white_{hemi}"],
            radius=radius,
            interpolation="linear",
        )
        hemis.append(surf_data)

    return np.vstack(hemis)


def main():
    parser = argparse.ArgumentParser(
        description="Project volumetric fMRI to fsaverage5 surface"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", type=str, help="Single NIfTI file to project")
    group.add_argument("--input-dir", type=str, help="Directory of NIfTI files")
    parser.add_argument(
        "--pattern", type=str, default="*.nii.gz", help="Glob pattern for --input-dir"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results/surface_projections",
        help="Output directory",
    )
    parser.add_argument(
        "--mesh", type=str, default="fsaverage5", help="Target mesh (default: fsaverage5)"
    )
    parser.add_argument(
        "--radius", type=float, default=3.0, help="Projection radius in mm (default: 3.0)"
    )
    parser.add_argument(
        "--save-mean", action="store_true", help="Also save the temporal mean projection"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.input:
        nifti_files = [Path(args.input)]
    else:
        nifti_files = sorted(Path(args.input_dir).glob(args.pattern))
        if not nifti_files:
            print(f"No files matching '{args.pattern}' in {args.input_dir}")
            return

    for nifti_path in nifti_files:
        print(f"Projecting: {nifti_path.name}")
        print(f"  Volume shape: {nib.load(str(nifti_path)).shape}")

        surf_data = project_volume_to_surface(nifti_path, mesh=args.mesh, radius=args.radius)
        print(f"  Surface shape: {surf_data.shape}  (vertices x timepoints)")

        out_name = nifti_path.stem.replace(".nii", "")
        out_path = output_dir / f"{out_name}_{args.mesh}.npy"
        np.save(out_path, surf_data)
        print(f"  Saved: {out_path}")

        if args.save_mean and surf_data.ndim == 2:
            mean_path = output_dir / f"{out_name}_{args.mesh}_mean.npy"
            np.save(mean_path, surf_data.mean(axis=1))
            print(f"  Saved mean: {mean_path}")

    print(f"\nAll projections saved to: {output_dir}")


if __name__ == "__main__":
    main()
