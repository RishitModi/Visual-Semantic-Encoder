"""Re-zips an unpacked PyTorch checkpoint folder back into a single .pth file.

Usage:
    python scripts\\repack_pth.py "<folder that contains data.pkl>" <output .pth file>
Example:
    python scripts\\repack_pth.py "C:\\Users\\you\\Downloads\\checkpoint_img_caption.pth" checkpoints_custom\\checkpoint_img_caption.pth
"""
import os
import sys
import zipfile


def find_root(folder):
    """Return the directory that directly contains data.pkl (handles one extra nesting level)."""
    for root, _dirs, files in os.walk(folder):
        if "data.pkl" in files:
            return root
    return None


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    folder, out = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])

    if not os.path.isdir(folder):
        sys.exit(f"ERROR: '{folder}' is not a folder.")
    root = find_root(folder)
    if root is None:
        sys.exit("ERROR: no data.pkl found inside that folder - it is not an unpacked checkpoint.")
    if os.path.exists(out):
        sys.exit(f"ERROR: '{out}' already exists. Delete it (or choose another name) and run again.")
    if os.path.commonpath([root, out]) == root:
        sys.exit("ERROR: the output file must be OUTSIDE the unpacked folder.")

    os.makedirs(os.path.dirname(out), exist_ok=True)
    prefix = os.path.splitext(os.path.basename(out))[0]   # torch expects one top-level folder name inside the zip

    entries = []
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            full = os.path.join(dirpath, name)
            entries.append((os.path.relpath(full, root).replace(os.sep, "/"), full))
    entries.sort(key=lambda e: (e[0] != "data.pkl", e[0]))   # data.pkl first, like torch.save does

    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as z:   # no compression, like torch.save
        for rel, full in entries:
            z.write(full, f"{prefix}/{rel}")

    print(f"Wrote {out}  ({os.path.getsize(out) / 1e6:.1f} MB, {len(entries)} files)")


if __name__ == "__main__":
    main()