"""Batch runner for FastHenry and FasterCap.

This script reads a list of geometry folders from an Address.txt file and
runs the existing ``run_fasthenry.py`` and ``run_fastercap.py`` utilities
for each entry. It expects each listed folder to contain a ``FastSolver``
subdirectory with the two generated solver inputs:

* ``Wire_Sections.inp`` (for FastHenry)
* ``Wire_Sections_FastCap.txt`` (for FasterCap)

The outputs ``Zc.mat`` and ``CapacitanceMatrix.txt`` are checked after each
run and warnings are printed if anything is missing.
"""
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from FastSolver.FastHenry.run_fasthenry import run_fasthenry
from FastSolver.FastCap.run_fastercap import run_fastercap


def normalize_address_path(raw_path: str | Path) -> Path:
    """Return a resolved ``Address.txt`` path from user input.

    The function accepts Windows-style paths (with backslashes and drive
    letters) and gracefully handles surrounding quotes copied from file
    explorers. If a directory is provided, an ``Address.txt`` inside that
    directory is used.
    """

    cleaned = str(raw_path).strip().strip("\"").strip("'")
    path = Path(cleaned).expanduser()

    if path.is_dir():
        candidate = path / "Address.txt"
        if candidate.is_file():
            path = candidate

    if not path.name.lower().endswith("address.txt") and not path.is_file():
        # Allow passing a directory-like string even if it doesn't currently
        # exist (e.g., when running on a different OS). Fall back to
        # appending Address.txt so downstream error messages make sense.
        path = path / "Address.txt"

    return path.resolve()


def read_address_lines(address_file: Path) -> list[str]:
    address_file = normalize_address_path(address_file)
    if not address_file.is_file():
        raise FileNotFoundError(f"Address.txt not found: {address_file}")

    with address_file.open("r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines()]

    return [line for line in lines if line]


def process_geometry_folder(base_path, eps_r: float = 1.0):
    geometry_root = os.path.abspath(base_path)
    fastsolver_dir = os.path.join(geometry_root, "FastSolver")

    if not os.path.isdir(fastsolver_dir):
        print(f"Warning: FastSolver folder missing for {geometry_root}")
        return

    fh_input = os.path.join(fastsolver_dir, "Wire_Sections.inp")
    fc_input = os.path.join(fastsolver_dir, "Wire_Sections_FastCap.txt")

    if not os.path.isfile(fh_input):
        print(f"Warning: Wire_Sections.inp missing in {fastsolver_dir}")
    else:
        print(f"Running FastHenry for {fh_input}")
        zc_path = run_fasthenry(fh_input)
        if os.path.isfile(zc_path):
            print(f"FastHenry output found: {zc_path}")
        else:
            print(f"Warning: Zc.mat not found after running FastHenry for {fh_input}")

    if not os.path.isfile(fc_input):
        print(f"Warning: Wire_Sections_FastCap.txt missing in {fastsolver_dir}")
    else:
        print(f"Running FasterCap for {fc_input} (eps_r = {eps_r})")
        # Note: we pass eps_r as a keyword, so 'options' keeps its default
        cap_path = run_fastercap(fc_input, eps_r=eps_r)
        if os.path.isfile(cap_path):
            print(f"FasterCap output found: {cap_path}")
        else:
            print(
                "Warning: CapacitanceMatrix.txt not found after running "
                f"FasterCap for {fc_input}"
            )



if __name__ == "__main__":
    if len(sys.argv) >= 2:
        address_txt = normalize_address_path(sys.argv[1])
    else:
        address_txt = normalize_address_path(input("Enter path to Address.txt: "))

    # NEW: parse optional eps_r argument
    if len(sys.argv) >= 3:
        try:
            eps_r = float(sys.argv[2])
        except ValueError:
            print("Invalid eps_r value. Example: 3.5 for FR-4-like dielectric.")
            sys.exit(1)
    else:
        eps_r = 1  # default: vacuum/air

    try:
        geometry_folders = read_address_lines(address_txt)
    except FileNotFoundError as exc:
        print(exc)
        sys.exit(1)

    if not geometry_folders:
        print("No geometry folders listed in Address.txt.")
        sys.exit(0)

    for folder in geometry_folders:
        print("\n=== Processing geometry folder ===")
        print(folder)
        process_geometry_folder(folder, eps_r=eps_r)

    print("\nBatch processing complete.")

