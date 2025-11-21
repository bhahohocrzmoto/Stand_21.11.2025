#!/usr/bin/env python3
"""
txt2dxf_sections.py

Reads a text file with lines like
  mm
  vol_res_cm=...,coil_res_cm=..., ...
  Section-1,25.50000000,0.00000000,0.00000000,1.00000000
  Section-1,25.50..., 0.2004..., 0.0000..., 1.00000000
  Section-2, ...

…where:
  - The **first meaningful line** is the **unit**: mm / cm / m (case-insensitive).
  - Lines starting with "Section-" contain: name, X, Y, Z, Current
    (Current is ignored for geometry).
  - Each unique "Section-N" is a **polyline** made of all its rows in file order.

Outputs a DXF with one polyline per Section.
If `ezdxf` is installed, we use it and set $INSUNITS.
If not, we write a minimal R12 DXF by hand (no dependencies).

Author: ChatGPT (Lumi) for Bahadir — with excessive comments by request :)
"""

from __future__ import annotations
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple

# ----------------------------- Parsing helpers --------------------------------

SectionName = str
Point3D = Tuple[float, float, float]


def detect_unit(lines: List[str]) -> str:
    """
    Find the first non-empty, non-comment token that is exactly 'mm', 'cm', or 'm'.
    If none is found, default to 'mm' (safer for PCB/coil drawings).
    """
    unit_pat = re.compile(r"^\s*(mm|cm|m)\s*$", re.IGNORECASE)
    for raw in lines:
        s = raw.strip()
        if not s:
            continue
        m = unit_pat.match(s)
        if m:
            return m.group(1).lower()
        # Allow header noise (e.g., 'vol_res_cm=...'), just skip
    return "mm"


def parse_sections(lines: List[str]) -> Dict[SectionName, List[Point3D]]:
    """
    Parse all 'Section-...' rows.
    Accepts flexible whitespace; values are comma-separated.
    Expected columns per row: SectionName, X, Y, Z, Current
    We **ignore** the last field (Current).

    Returns:
        dict: section_name -> list of (x, y, z) in **file order**.
    """
    sections: Dict[SectionName, List[Point3D]] = {}
    for raw in lines:
        s = raw.strip()
        if not s or s.lower() in {"mm", "cm", "m"}:
            continue
        if not s.startswith("Section-"):
            # skip header lines like "vol_res_cm=...,coil_res_cm=...,margin_cm=...,box=auto"
            continue

        # Split by comma; allow spaces around commas
        parts = [p.strip() for p in s.split(",")]
        if len(parts) < 5:
            # Be explicit to help debugging input issues
            sys.stderr.write(f"[warn] Skipping malformed line (needs 5 fields): {s}\n")
            continue

        sec = parts[0]  # e.g., 'Section-1'
        try:
            x = float(parts[1])
            y = float(parts[2])
            z = float(parts[3])
            # parts[4] = current -> ignored
        except ValueError:
            sys.stderr.write(f"[warn] Skipping non-numeric line: {s}\n")
            continue

        sections.setdefault(sec, []).append((x, y, z))

    if not sections:
        raise ValueError("No 'Section-*' lines were found. Check the file format.")
    return sections


def any_nonzero_z(points: List[Point3D]) -> bool:
    """Return True if any vertex has |z| > tiny threshold."""
    eps = 1e-15
    return any(abs(p[2]) > eps for p in points)


# ------------------------------ DXF writers -----------------------------------

def write_with_ezdxf(
    out_path: Path,
    sections: Dict[SectionName, List[Point3D]],
    unit: str,
) -> None:
    """
    Write the DXF using ezdxf if available.
    - Sets $INSUNITS so downstream CAD knows the unit.
    - Uses LWPOLYLINE for 2D sections (all z==0) and POLYLINE3D when z≠0 exists.
    """
    import ezdxf  # type: ignore

    # R2010 is a good default many viewers open happily
    doc = ezdxf.new(dxfversion="R2010")
    msp = doc.modelspace()

    # Map string unit to DXF $INSUNITS enum
    insunits_map = {"in": 1, "ft": 2, "mi": 3, "mm": 4, "cm": 5, "m": 6, "km": 7}
    doc.header["$INSUNITS"] = insunits_map.get(unit, 4)

    # Create one layer per section for clarity (optional, helps toggling visibility)
    for sec_name, pts in sections.items():
        layer = sec_name  # e.g., "Section-1"
        if layer not in doc.layers:
            doc.layers.add(layer)

        if any_nonzero_z(pts):
            # 3D polyline
            msp.add_polyline3d(points=pts, dxfattribs={"layer": layer})
        else:
            # 2D lightweight polyline (faster, smaller DXF)
            xy = [(x, y) for x, y, _ in pts]
            # NOT closed by default; change close=True if sections should be looped
            msp.add_lwpolyline(xy, format="xy", dxfattribs={"layer": layer, "const_width": 0.0})

    # Stamp a tiny note with the unit in the corner (handy when INSUNITS is ignored by some viewers)
    try:
        # Put note at min corner of all points
        all_xy = [(x, y) for pts in sections.values() for (x, y, _) in pts]
        minx = min(p[0] for p in all_xy)
        miny = min(p[1] for p in all_xy)
        msp.add_mtext(f"Unit: {unit.upper()}  •  Sections: {len(sections)}").set_location((minx, miny))
    except Exception:
        pass

    doc.saveas(out_path.as_posix())


def write_r12_minimal(
    out_path: Path,
    sections: Dict[SectionName, List[Point3D]],
    unit: str,
) -> None:
    """
    Minimalist, dependency-free **DXF R12** writer.
    - Writes each Section as a classic POLYLINE with 3D vertices (works for z=0 too).
    - R12 has no $INSUNITS; we insert a small TEXT label “Unit: ...” near the min corner.
    - Very small and robust; opens in AutoCAD/LibreCAD/Draftsight/etc.

    NOTE: Classic POLYLINE cannot carry explicit layer colors without more header setup;
          we just set the layer name to the section name.
    """
    # Compute a rough label position
    try:
        all_xy = [(x, y) for pts in sections.values() for (x, y, _) in pts]
        minx = min(p[0] for p in all_xy)
        miny = min(p[1] for p in all_xy)
    except Exception:
        minx, miny = (0.0, 0.0)

    def _pair(code: str | int, value: str | float | int) -> str:
        """DXF group code pair formatter."""
        return f"{code}\n{value}\n"

    # Build DXF as string (R12 structure)
    chunks: List[str] = []
    # --- HEADER ---
    chunks.append("0\nSECTION\n2\nHEADER\n0\nENDSEC\n")
    # --- TABLES (define layers, one per section) ---
    chunks.append("0\nSECTION\n2\nTABLES\n")
    # LAYER table
    chunks.append("0\nTABLE\n2\nLAYER\n70\n{}\n".format(len(sections)))
    for layer_name in sections.keys():
        chunks.append("0\nLAYER\n2\n{}\n70\n0\n62\n7\n6\nCONTINUOUS\n".format(layer_name))
    chunks.append("0\nENDTAB\n0\nENDSEC\n")
    # --- ENTITIES ---
    chunks.append("0\nSECTION\n2\nENTITIES\n")

    # Add a tiny TEXT entity as a unit note
    chunks.append("0\nTEXT\n8\n{}\n10\n{}\n20\n{}\n30\n0\n40\n3.5\n1\n{}\n".format(
        "NOTES", minx, miny, f"Unit: {unit.upper()}  Sections: {len(sections)}")
    )

    for layer_name, pts in sections.items():
        if len(pts) < 2:
            continue  # need at least a segment to see anything

        # Start POLYLINE (3D)
        chunks.append("0\nPOLYLINE\n8\n{}\n66\n1\n70\n8\n".format(layer_name))  # 66=has vertices, 70=3D poly
        # VERTEX records
        for (x, y, z) in pts:
            chunks.append("0\nVERTEX\n8\n{}\n10\n{}\n20\n{}\n30\n{}\n".format(layer_name, x, y, z))
        # End of the sequence
        chunks.append("0\nSEQEND\n")

    # End ENTITIES and file
    chunks.append("0\nENDSEC\n0\nEOF\n")

    out_path.write_text("".join(chunks), encoding="ascii")


# ------------------------------ Main program ----------------------------------

def main():
    # 1) Resolve input path (arg or default)
    if len(sys.argv) > 1:
        in_path = Path(sys.argv[1]).expanduser()
    else:
        # Default to a friendly name if user double-clicks the script
        in_path = Path("Wire_Sections.txt")

    if not in_path.exists():
        sys.stderr.write(f"[error] Input file not found: {in_path}\n")
        sys.stderr.write("Usage: python txt2dxf_sections.py /path/to/Wire_Sections.txt\n")
        sys.exit(1)

    raw_lines = in_path.read_text(encoding="utf-8", errors="ignore").splitlines()

    # 2) Detect unit & parse geometry
    unit = detect_unit(raw_lines)  # 'mm' / 'cm' / 'm'
    sections = parse_sections(raw_lines)

    # 3) Decide DXF writer
    out_path = in_path.with_suffix(".dxf")
    try:
        import importlib.util
        has_ezdxf = importlib.util.find_spec("ezdxf") is not None
    except Exception:
        has_ezdxf = False

    if has_ezdxf:
        print(f"[info] Using ezdxf writer (unit={unit}).")
        write_with_ezdxf(out_path, sections, unit)
    else:
        print(f"[info] ezdxf not found → writing minimal R12 DXF (unit='{unit}' noted as TEXT).")
        print("        Install ezdxf for nicer output:  pip install ezdxf")
        write_r12_minimal(out_path, sections, unit)

    # 4) Small report
    n_pts = sum(len(v) for v in sections.values())
    print(f"[ok] Wrote {out_path}  •  sections={len(sections)}  •  vertices={n_pts}  •  unit={unit}")

if __name__ == "__main__":
    main()
