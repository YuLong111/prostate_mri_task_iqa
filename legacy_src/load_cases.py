import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib


def load_nifti(path: str):
    img = nib.load(path)
    data = img.get_fdata(dtype=np.float32)
    return data, img.affine, img.header.get_zooms()


def show_mid_slices(vol: np.ndarray, title: str):
    """Show mid axial slice, rotated 90° anticlockwise for display."""
    z = vol.shape[2] // 2
    img2d = np.rot90(vol[:, :, z], k=3)  # 90° anticlockwise
    plt.imshow(img2d, cmap="gray")
    plt.title(title)
    plt.axis("off")


def main():
    # EDIT THESE PATHS
    dwi_path  = r"D:\1\杂物\学校\ucl\year3\project\project\OneDrive_1_2026-1-15\dwi\Patient050630693_study_0.nii.gz"
    t2_path   = r"D:\1\杂物\学校\ucl\year3\project\project\OneDrive_1_2026-1-15\t2\Patient050630693_study_0.nii.gz"
    mask_path = r"D:\1\杂物\学校\ucl\year3\project\project\OneDrive_1_2026-1-15\prostate_mask\Patient050630693_study_0.nii.gz"

    dwi, dwi_aff, dwi_zooms = load_nifti(dwi_path)
    t2,  t2_aff,  t2_zooms  = load_nifti(t2_path)
    msk, msk_aff, msk_zooms = load_nifti(mask_path)

    print("DWI shape/zooms:", dwi.shape, dwi_zooms)
    print("T2  shape/zooms:", t2.shape,  t2_zooms)
    print("MSK shape/zooms:", msk.shape, msk_zooms)

    print("Affine close (DWI vs T2):", np.allclose(dwi_aff, t2_aff, atol=1e-3))
    print("Affine close (MSK vs T2):", np.allclose(msk_aff, t2_aff, atol=1e-3))

    # Use the same mid-slice index for overlay (assumes volumes aligned / same depth)
    z = t2.shape[2] // 2

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    show_mid_slices(dwi, "DWI mid-slice (rotated)")

    plt.subplot(1, 3, 2)
    show_mid_slices(t2, "T2 mid-slice (rotated)")

    plt.subplot(1, 3, 3)
    dwi2d = np.rot90(dwi[:, :, z], k=1)          # rotate for display
    msk2d = np.rot90((msk[:, :, z] > 0), k=3)    # rotate mask same way

    plt.imshow(dwi2d, cmap="gray")
    plt.imshow(msk2d, alpha=0.3)
    plt.title("DWI + mask overlay (rotated)")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()