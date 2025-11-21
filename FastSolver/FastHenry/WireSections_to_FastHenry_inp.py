"""Utility to convert FreeCAD wire section exports into FastHenry input files.

The script reads the ``Wire_Sections.txt`` file that FreeCAD's ElectroMagnetic
Workbench produces and generates a ``fasthenry_input_file.inp`` file containing
nodes, segments and ports for FastHenry 2.

Example
-------
    python convert_wire_sections.py Wire_Sections.txt fasthenry_input_file.inp

Use ``--help`` to see all of the optional parameters (segment width/height,
default material parameters, frequency sweep, ...).
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from pathlib import Path
from typing import List, Tuple


SectionPoints = List[Tuple[float, float, float]]


def parse_wire_sections(path: Path) -> Tuple[str, str | None, OrderedDict[str, SectionPoints]]:
    """Parse the FreeCAD ``Wire_Sections.txt`` file.

    The file uses the following structure::

        <units>
        <metadata key=value pairs>

        Section-1,x,y,z,extra
        Section-1,x,y,z,extra
        Section-2,x,y,z,extra
        ...

    Only the first three numeric columns are needed to build the FastHenry
    geometry, so the remaining data on each row is ignored but tolerated.
    """

    raw_lines = [line.strip() for line in path.read_text().splitlines()]
    idx = 0

    def _consume_non_empty(start: int) -> int:
        while start < len(raw_lines) and not raw_lines[start]:
            start += 1
        return start

    idx = _consume_non_empty(idx)
    if idx >= len(raw_lines):
        raise ValueError("Wire_Sections file is empty.")

    units = raw_lines[idx]
    idx += 1

    idx = _consume_non_empty(idx)
    metadata = None
    if idx < len(raw_lines) and "=" in raw_lines[idx]:
        metadata = raw_lines[idx]
        idx += 1

    idx = _consume_non_empty(idx)

    sections: "OrderedDict[str, SectionPoints]" = OrderedDict()
    for line in raw_lines[idx:]:
        if not line:
            continue
        if line.startswith("#"):
            # Allow comments for robustness.
            continue
        label, *values = [chunk.strip() for chunk in line.split(",")]
        if len(values) < 3:
            raise ValueError(f"Malformed line (expected at least 3 numeric columns): {line!r}")
        try:
            x, y, z = (float(values[0]), float(values[1]), float(values[2]))
        except ValueError as exc:  # pragma: no cover - defensive.
            raise ValueError(f"Unable to parse coordinates from line: {line!r}") from exc

        sections.setdefault(label, []).append((x, y, z))

    if not sections:
        raise ValueError("No section data found in Wire_Sections file.")

    return units, metadata, sections


def _format_float(value: float) -> str:
    """Match the formatting style used by the reference FastHenry file."""

    # ``:.8f`` keeps enough precision for FreeCAD's output. The rstrip calls
    # remove unneeded zeros while keeping a single leading zero for small values.
    text = f"{value:.8f}"
    if "e" in text or "E" in text:
        return text
    text = text.rstrip("0").rstrip(".")
    if text == "-0":  # avoid ``-0`` which FastHenry dislikes.
        text = "0"
    if text == "":
        text = "0"
    if "." not in text and "e" not in text.lower():
        text = f"{text}.0"
    if text[0] == ".":
        text = "0" + text
    if text.startswith("-."):
        text = text.replace("-.", "-0.", 1)
    return text


def build_inp_content(
    *,
    units: str,
    metadata: str | None,
    sections: OrderedDict[str, SectionPoints],
    segment_width: float,
    segment_height: float,
    sigma: float,
    nhinc: int,
    nwinc: int,
    rh: float,
    rw: float,
    freq_min: float,
    freq_max: float,
    ndec: float,
) -> str:
    lines: List[str] = []
    lines.append("* FastHenry input file created using FreeCAD's ElectroMagnetic Workbench")
    lines.append("* See http://www.freecad.org, http://www.fastfieldsolvers.com and http://epc-co.com")
    if metadata:
        lines.append(f"* Source metadata: {metadata}")
    lines.append("")

    if units.lower().startswith(".units"):
        lines.append(units)
    else:
        lines.append(f".units {units}")
    lines.append("")

    lines.append(
        f".default sigma={sigma} nhinc={nhinc} nwinc={nwinc} rh={rh} rw={rw}"
    )
    lines.append("")

    lines.append("* Nodes")
    for section_name, points in sections.items():
        for index, (x, y, z) in enumerate(points, start=1):
            node_name = f"N{section_name}_Node_{index}"
            lines.append(
                f"{node_name} x={_format_float(x)} y={_format_float(y)} z={_format_float(z)}"
            )
    lines.append("")

    lines.append("* Segments")
    segment_counter = 0
    for section_name, points in sections.items():
        for index in range(1, len(points)):
            start_node = f"N{section_name}_Node_{index}"
            end_node = f"N{section_name}_Node_{index + 1}"
            if segment_counter == 0:
                seg_name = "EFHSegment"
            else:
                seg_name = f"EFHSegment{segment_counter:03d}"
            lines.append(
                f"{seg_name} {start_node} {end_node} w={_format_float(segment_width)} h={_format_float(segment_height)}"
            )
            segment_counter += 1
    lines.append("")

    lines.append("* Ports")
    for section_name, points in sections.items():
        start_node = f"N{section_name}_Node_1"
        end_node = f"N{section_name}_Node_{len(points)}"
        lines.append(f".external {start_node} {end_node}")
    lines.append("")

    lines.append(
        f".freq fmin={freq_min} fmax={freq_max} ndec={ndec}"
    )
    lines.append("")
    lines.append(".end")

    return "\n".join(lines) + "\n"


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Path to Wire_Sections.txt")
    parser.add_argument("output", type=Path, help="Destination FastHenry .inp file")
    parser.add_argument("--segment-width", type=float, default=0.25, help="Segment width (w)")
    parser.add_argument("--segment-height", type=float, default=0.035, help="Segment height (h)")
    parser.add_argument("--sigma", type=float, default=58000.0, help="Conductor conductivity")
    parser.add_argument("--nhinc", type=int, default=1, help="FastHenry nhinc parameter")
    parser.add_argument("--nwinc", type=int, default=1, help="FastHenry nwinc parameter")
    parser.add_argument("--rh", type=float, default=2.0, help="FastHenry rh parameter")
    parser.add_argument("--rw", type=float, default=2.0, help="FastHenry rw parameter")
    parser.add_argument("--fmin", type=float, default=1.0, help="Minimum frequency")
    parser.add_argument(
        "--fmax", type=float, default=1_000_000_000.0, help="Maximum frequency"
    )
    parser.add_argument("--ndec", type=float, default=1.0, help="Frequency points per decade")
    return parser


def main(args: argparse.Namespace | None = None) -> None:
    parser = build_argument_parser()
    if args is None:
        args = parser.parse_args()

    units, metadata, sections = parse_wire_sections(args.input)
    content = build_inp_content(
        units=units,
        metadata=metadata,
        sections=sections,
        segment_width=args.segment_width,
        segment_height=args.segment_height,
        sigma=args.sigma,
        nhinc=args.nhinc,
        nwinc=args.nwinc,
        rh=args.rh,
        rw=args.rw,
        freq_min=args.fmin,
        freq_max=args.fmax,
        ndec=args.ndec,
    )

    args.output.write_text(content)


if __name__ == "__main__":
    main()
