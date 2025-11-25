"""Interactive helper that batch-converts Wire_Sections.txt files for FastSolver tools.

The script asks the user for the common FastHenry / FastCap parameters and then
creates a ``FastSolver`` sub-folder next to every ``Wire_Sections.txt`` entry
listed inside an ``Address.txt`` file. Both FastHenry (``.inp``) and FastCap
(``_FastCap.txt``) files are generated for each entry.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
import sys
from typing import Iterable, List

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from FastSolver.FastCap import WireSections_to_FastCap_txt as fastcap
from FastSolver.FastHenry import WireSections_to_FastHenry_inp as fasthenry


DEFAULT_SEGMENT_WIDTH = 0.25
DEFAULT_SEGMENT_HEIGHT = 0.035
DEFAULT_SIGMA = 58_000.0
DEFAULT_FMIN = 1_000.0
DEFAULT_FMAX = 1_000_000.0
DEFAULT_NDEC = 1.0
DEFAULT_TRACE_WIDTH_MM = 0.25


@dataclass(slots=True)
class ConversionSettings:
    segment_width: float = DEFAULT_SEGMENT_WIDTH
    segment_height: float = DEFAULT_SEGMENT_HEIGHT
    sigma: float = DEFAULT_SIGMA
    fmin: float = DEFAULT_FMIN
    fmax: float = DEFAULT_FMAX
    ndec: float = DEFAULT_NDEC
    trace_width_mm: float = DEFAULT_TRACE_WIDTH_MM
    nhinc: int = 1
    nwinc: int = 1
    rh: float = 2.0
    rw: float = 2.0


def _prompt_float(label: str, default: float) -> float:
    while True:
        raw = input(f"{label} [{default}]: ").strip()
        if not raw:
            return default
        try:
            return float(raw.replace(",", "."))
        except ValueError:
            print("Please enter a valid number.")


def _prompt_path(label: str) -> Path:
    while True:
        raw = input(f"{label}: ").strip()
        if raw:
            return Path(raw.strip("\"").strip("'"))


def _normalize_address_file(raw: Path | str) -> Path:
    """Resolve a user-supplied path to the Address.txt file.

    Handles Windows-style paths (with drive letters/backslashes) and strips any
    surrounding quotes pasted from file explorers. If the provided path points
    to a directory, the ``Address.txt`` inside that directory is used.
    """

    cleaned = str(raw).strip().strip("\"").strip("'")
    path = Path(cleaned).expanduser()

    if path.is_dir():
        candidate = path / "Address.txt"
        if candidate.is_file():
            path = candidate

    if not path.name.lower().endswith("address.txt") and not path.is_file():
        path = path / "Address.txt"

    return path.resolve()


def _resolve_directory(raw: str, address_file: Path) -> Path:
    raw = raw.strip()
    base_dir = address_file.parent

    candidates: List[Path] = []
    if raw:
        candidates.append(Path(raw).expanduser())
        candidates.append((base_dir / raw).expanduser())

        if ":" in raw or "\\" in raw:
            windows_path = PureWindowsPath(raw)
            if "Example" in windows_path.parts:
                idx = windows_path.parts.index("Example")
                relative = Path(*windows_path.parts[idx + 1 :])
                candidates.append(base_dir / relative)
            if len(windows_path.parts) > 1:
                candidates.append(Path(*windows_path.parts[1:]))

    for candidate in candidates:
        if candidate.is_dir():
            return candidate

    # Fall back to the first candidate for error reporting.
    return candidates[0] if candidates else base_dir


def _iter_wire_directories(address_file: Path) -> Iterable[Path]:
    for line in address_file.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        directory = _resolve_directory(line, address_file)
        if not directory.is_dir():
            print(f"[WARN] Skipping '{line}' – directory not found ({directory}).")
            continue
        yield directory


def _gather_settings(args: argparse.Namespace) -> ConversionSettings:
    interactive = not args.non_interactive
    settings = ConversionSettings()

    def choose(value, default, label):
        if value is not None:
            return value
        if not interactive:
            return default
        return _prompt_float(label, default)

    settings.segment_width = choose(
        args.segment_width, settings.segment_width, "Segment width in mm"
    )
    settings.segment_height = choose(
        args.segment_height, settings.segment_height, "Segment height in mm"
    )
    settings.sigma = choose(args.sigma, settings.sigma, "Sigma (S/m)")
    settings.fmin = choose(args.fmin, settings.fmin, "fmin (Hz)")
    settings.fmax = choose(args.fmax, settings.fmax, "fmax (Hz)")
    settings.ndec = choose(args.ndec, settings.ndec, "Points per decade (ndec)")
    settings.trace_width_mm = choose(
        args.trace_width_mm, settings.trace_width_mm, "Trace width for FastCap (mm)"
    )
    return settings


def _convert_directory(directory: Path, settings: ConversionSettings) -> None:
    wire_sections = directory / "Wire_Sections.txt"
    if not wire_sections.is_file():
        print(f"[WARN] '{directory}' does not contain Wire_Sections.txt – skipped.")
        return

    output_dir = directory / "FastSolver"
    output_dir.mkdir(exist_ok=True)

    fasthenry_output = output_dir / f"{wire_sections.stem}.inp"
    fastcap_output = output_dir / f"{wire_sections.stem}_FastCap.txt"

    units, metadata, sections = fasthenry.parse_wire_sections(wire_sections)
    fasthenry_output.write_text(
        fasthenry.build_inp_content(
            units=units,
            metadata=metadata,
            sections=sections,
            segment_width=settings.segment_width,
            segment_height=settings.segment_height,
            sigma=settings.sigma,
            nhinc=settings.nhinc,
            nwinc=settings.nwinc,
            rh=settings.rh,
            rw=settings.rw,
            freq_min=settings.fmin,
            freq_max=settings.fmax,
            ndec=settings.ndec,
        )
    )

    _, cap_sections = fastcap.parse_wire_sections(wire_sections)
    fastcap.write_fastcap_file(fastcap_output, cap_sections, settings.trace_width_mm)

    # Use ASCII arrow to avoid UnicodeEncodeError on Windows console codepages
    print(f"[OK] Converted {directory.name} -> {fasthenry_output.name}, {fastcap_output.name}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Interactive batch converter for FastHenry/FastCap input files."
    )
    parser.add_argument(
        "address_file",
        nargs="?",
        type=Path,
        help="Path to Address.txt (list of directories containing Wire_Sections.txt)",
    )
    parser.add_argument("--segment-width", type=float, help="Segment width (mm)")
    parser.add_argument("--segment-height", type=float, help="Segment height (mm)")
    parser.add_argument("--sigma", type=float, help="Conductivity sigma (S/m)")
    parser.add_argument("--fmin", type=float, help="Minimum frequency (Hz)")
    parser.add_argument("--fmax", type=float, help="Maximum frequency (Hz)")
    parser.add_argument("--ndec", type=float, help="Points per decade")
    parser.add_argument(
        "--trace-width-mm", type=float, help="Trace width for FastCap panels (mm)"
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Use defaults/CLI options without prompting.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    address_file: Path
    if args.address_file is None:
        if args.non_interactive:
            raise SystemExit("Address file must be provided when --non-interactive is used.")
        address_file = _prompt_path("Path to Address.txt")
    else:
        address_file = args.address_file

    address_file = _normalize_address_file(address_file)
    if not address_file.is_file():
        raise SystemExit(f"Address file not found: {address_file}")

    settings = _gather_settings(args)

    directories = list(_iter_wire_directories(address_file))
    if not directories:
        print("No valid directories found in address file.")
        return

    for directory in directories:
        _convert_directory(directory, settings)


if __name__ == "__main__":
    main()

