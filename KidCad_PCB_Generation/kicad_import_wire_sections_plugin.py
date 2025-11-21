# -*- coding: utf-8 -*-
"""
Import Wire Sections — KiCad Action Plugin (KiCad 6/7/8/9)
=========================================================

v9c highlights
--------------
• Parses Z for each Section and **maps unique Z-levels to copper layers**:
  - Sort Z ascending → bottom to top (smallest Z → B.Cu, largest Z → F.Cu).
  - Auto-sets the board **copper layer count** to match number of Z-levels.
  - Places sections on the correct layer (per their Z).
• Keeps a **single-layer mode** if you prefer (same as earlier versions).
• Optional: **save a copy of the board** into each source folder that had a
  Wire_Sections.txt (filename pattern explained below).

Z grouping
----------
Z values are read from each "Section-#, x, y, z, I_amp" row.
All vertices of a Section are assumed to share the same Z (we take the first
encountered Z for that Section). Z units follow the file's unit line (e.g. "mm").
To be robust against tiny numerical noise, Z-values are grouped with a tolerance
of 1e-6 mm.

File format reminder
--------------------
Wire_Sections.txt
    1st non-empty line → units (e.g. "mm", "um", "inch", "mil", ...)
    next line(s)      → header (ignored for layout)
    rest              → "Section-#, x, y, z, I_amp"
                        (z & I_amp used only for layer mapping; I_amp ignored)

Author: ChatGPT (Lumi) for Bahadir
"""

import os
from typing import Dict, List, Tuple, Optional

import pcbnew
import wx

# ----- Compatibility helpers -------------------------------------------------

def _have(name: str) -> bool:
    return hasattr(pcbnew, name)


def _new_track(board: "pcbnew.BOARD"):
    cls = getattr(pcbnew, "PCB_TRACK", None) or getattr(pcbnew, "TRACK", None)
    if cls is None:
        raise RuntimeError("No pcbnew.PCB_TRACK or pcbnew.TRACK in this KiCad build.")
    return cls(board)


def _set_track_points(seg, x1_mm: float, y1_mm: float, x2_mm: float, y2_mm: float):
    """Try wxPointMM first (coil-style), then VECTOR2I, then wxPoint(IU)."""
    try:
        seg.SetStart(pcbnew.wxPointMM(x1_mm, y1_mm))
        seg.SetEnd(  pcbnew.wxPointMM(x2_mm, y2_mm))
        return
    except Exception:
        pass
    if _have("VECTOR2I"):
        seg.SetStart(pcbnew.VECTOR2I(pcbnew.FromMM(x1_mm), pcbnew.FromMM(y1_mm)))
        seg.SetEnd(  pcbnew.VECTOR2I(pcbnew.FromMM(x2_mm), pcbnew.FromMM(y2_mm)))
        return
    seg.SetStart(pcbnew.wxPoint(pcbnew.FromMM(x1_mm), pcbnew.FromMM(y1_mm)))
    seg.SetEnd(  pcbnew.wxPoint(pcbnew.FromMM(x2_mm), pcbnew.FromMM(y2_mm)))


def _set_no_net(seg):
    if hasattr(seg, "SetNetCode"):
        try:
            seg.SetNetCode(0)
            return
        except Exception:
            pass
    if hasattr(seg, "SetNet"):
        try:
            seg.SetNet(None)
        except Exception:
            pass


def _set_copper_layer_count(board: "pcbnew.BOARD", n_layers: int):
    """Set board copper layer count (KiCad 6/7/8/9)."""
    ds = board.GetDesignSettings()
    if hasattr(ds, "SetCopperLayerCount"):
        ds.SetCopperLayerCount(n_layers)
    else:
        # Older fallback (rare now)
        board.SetCopperLayerCount(n_layers)


# ----- Units -----------------------------------------------------------------

def _unit_scale(unit_str: str) -> float:
    u = (unit_str or "").strip().lower()
    if u in ("mm",):
        return 1.0
    if u in ("cm",):
        return 10.0
    if u in ("um", "µm", "micron", "microns"):
        return 1e-3
    if u in ("mil", "mils"):
        return 0.0254
    if u in ("in", "inch", "inches"):
        return 25.4
    return 1.0


# ----- Data ------------------------------------------------------------------

Point = Tuple[float, float]  # (x_mm, y_mm)

class SectionData:
    __slots__ = ("z_mm", "points")
    def __init__(self, z_mm: float):
        self.z_mm: float = z_mm
        self.points: List[Point] = []


def parse_wire_sections(txt_path: str):
    """
    Returns: (unit_str, scale_to_mm, sections: dict[name -> SectionData])
    Sections store z_mm and polyline points (x_mm, y_mm).
    """
    if not os.path.isfile(txt_path):
        raise FileNotFoundError(txt_path)

    unit_str = None
    sections: Dict[str, SectionData] = {}

    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        # unit line
        for raw in f:
            line = raw.strip()
            if line:
                unit_str = line
                break
        if unit_str is None:
            raise ValueError("Missing unit line (e.g., 'mm').")

        scale = _unit_scale(unit_str)

        # skip header until first Section- line
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("Section-"):
                _add_vertex(line, sections, scale)
                break

        # rest
        for raw in f:
            line = raw.strip()
            if not line or not line.startswith("Section-"):
                continue
            _add_vertex(line, sections, scale)

    return unit_str, scale, sections


def _add_vertex(line: str, sections: Dict[str, SectionData], scale: float):
    # Expect "Section-#, x, y, z, I_amp"
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 4:
        return
    sec_name = parts[0]
    try:
        x_mm = float(parts[1]) * scale
        y_mm = float(parts[2]) * scale
        z_mm = float(parts[3]) * scale
    except Exception:
        return
    if sec_name not in sections:
        sections[sec_name] = SectionData(z_mm=z_mm)
    # If z differs slightly later, we keep first z but you could check here.
    sections[sec_name].points.append((x_mm, y_mm))


# ----- Z → Layers mapping ----------------------------------------------------

def unique_sorted_z(sections: Dict[str, SectionData], eps_mm: float = 1e-6) -> List[float]:
    """Return unique Zs sorted ascending with tolerance (group nearly-equal)."""
    zs = sorted(s.z_mm for s in sections.values())
    unique: List[float] = []
    for z in zs:
        if not unique or abs(z - unique[-1]) > eps_mm:
            unique.append(z)
    return unique


def copper_layer_names_top_to_bottom(n_layers: int) -> List[str]:
    """['F.Cu', 'In1.Cu', ..., 'B.Cu'] for n_layers >= 2."""
    if n_layers <= 1:
        return ["F.Cu"]
    names = ["F.Cu"]
    for i in range(1, n_layers - 1):
        names.append(f"In{i}.Cu")
    names.append("B.Cu")
    return names


def build_z_to_layer_map(board: "pcbnew.BOARD", z_levels: List[float]) -> Dict[float, str]:
    """
    Given sorted ascending Zs, set copper layer count and return a dict z→layer_name.
    Mapping: smallest Z → B.Cu, largest Z → F.Cu, middle Zs → inner layers from bottom up.
    """
    n = max(2, len(z_levels))  # at least 2 copper layers if you have multiple Zs
    if len(z_levels) == 1:
        # Single z-level: still allow single layer boards (F.Cu) if user wants; but we map to F.Cu by default.
        n = 1

    _set_copper_layer_count(board, n)

    names_top_to_bottom = copper_layer_names_top_to_bottom(n)   # e.g., ['F.Cu', 'In1.Cu', ..., 'B.Cu']
    names_bottom_to_top = list(reversed(names_top_to_bottom))   # e.g., ['B.Cu', ..., 'F.Cu']

    z_to_layer: Dict[float, str] = {}
    for idx, z in enumerate(z_levels):
        # Clamp idx to available layers (just in case)
        layer_name = names_bottom_to_top[min(idx, len(names_bottom_to_top) - 1)]
        z_to_layer[z] = layer_name
    return z_to_layer


# ----- Drawing ---------------------------------------------------------------

def draw_sections(board: "pcbnew.BOARD",
                  sections: Dict[str, SectionData],
                  mode: str,
                  track_width_mm: float,
                  fixed_layer_name: Optional[str] = None,
                  collect_segments: bool = False) -> Tuple[int, Dict[str, int], List["pcbnew.PCB_TRACK"]]:
    """
    mode: 'single' or 'zmap'
    If 'single', fixed_layer_name must be given and all Sections go there.
    If 'zmap', Z levels are mapped bottom→top to B.Cu/In?.Cu/F.Cu.
    Returns: (segments_created, per_layer_counts)
    """
    iu_per_mm = getattr(pcbnew, "IU_PER_MM", 1_000_000)
    width_iu = int(track_width_mm * iu_per_mm)
    tol = 1e-9

    per_layer_counts: Dict[str, int] = {}
    created = 0
    created_segments: List["pcbnew.PCB_TRACK"] = []

    if mode == "single":
        layer_id = board.GetLayerID(fixed_layer_name)
        for sec_name, sec in sorted(sections.items()):
            pts = sec.points
            if len(pts) < 2:
                continue
            for (x1, y1), (x2, y2) in zip(pts[:-1], pts[1:]):
                if abs(x2 - x1) <= tol and abs(y2 - y1) <= tol:
                    continue
                seg = _new_track(board)
                _set_track_points(seg, x1, y1, x2, y2)
                seg.SetWidth(width_iu)
                seg.SetLayer(layer_id)
                _set_no_net(seg)
                board.Add(seg)
                created += 1
                if collect_segments:
                    created_segments.append(seg)
        per_layer_counts[fixed_layer_name] = created
        try: pcbnew.Refresh()
        except Exception: pass
        return created, per_layer_counts, created_segments

    # zmap mode
    z_levels = unique_sorted_z(sections)
    z_to_layer = build_z_to_layer_map(board, z_levels)  # also sets copper layer count

    # Prepare layer IDs
    layer_ids: Dict[str, int] = {name: board.GetLayerID(name) for name in set(z_to_layer.values())}

    for sec_name, sec in sorted(sections.items()):
        layer_name = z_to_layer[_snap_to_existing(sec.z_mm, z_levels)]
        layer_id = layer_ids[layer_name]
        cnt = 0

        pts = sec.points
        if len(pts) < 2:
            continue

        for (x1, y1), (x2, y2) in zip(pts[:-1], pts[1:]):
            if abs(x2 - x1) <= tol and abs(y2 - y1) <= tol:
                continue
            seg = _new_track(board)
            _set_track_points(seg, x1, y1, x2, y2)
            seg.SetWidth(width_iu)
            seg.SetLayer(layer_id)
            _set_no_net(seg)
            board.Add(seg)
            created += 1
            cnt += 1
            if collect_segments:
                created_segments.append(seg)

        per_layer_counts[layer_name] = per_layer_counts.get(layer_name, 0) + cnt

    try: pcbnew.Refresh()
    except Exception: pass

    return created, per_layer_counts, created_segments


def _snap_to_existing(z: float, z_levels: List[float], eps_mm: float = 1e-6) -> float:
    """Return the canonical z-level in z_levels that is within eps of z (first match)."""
    for z0 in z_levels:
        if abs(z - z0) <= eps_mm:
            return z0
    # If nothing within eps, choose nearest
    return min(z_levels, key=lambda v: abs(v - z))


# ----- UI helpers ------------------------------------------------------------

def ask_file_open(parent: wx.Window, title: str, wildcard: str) -> Optional[str]:
    with wx.FileDialog(parent, message=title, wildcard=wildcard, style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as dlg:
        if dlg.ShowModal() == wx.ID_OK:
            return dlg.GetPath()
    return None


def ask_track_width(parent: wx.Window, default_mm: float = 0.25) -> Optional[float]:
    with wx.TextEntryDialog(parent, "Track width (mm):", "Import Wire Sections", value=str(default_mm)) as dlg:
        if dlg.ShowModal() == wx.ID_OK:
            try:
                v = float(dlg.GetValue())
                if v <= 0:
                    raise ValueError
                return v
            except Exception:
                wx.MessageBox("Please enter a positive number for the width.", "Input", wx.OK | wx.ICON_WARNING)
                return None
    return None


def ask_mode(parent: wx.Window) -> Optional[str]:
    choices = ["Map Z → Layers automatically", "Single fixed layer (like before)"]
    with wx.SingleChoiceDialog(parent, "Import mode:", "Import Wire Sections", choices) as dlg:
        dlg.SetSelection(0)
        if dlg.ShowModal() == wx.ID_OK:
            return "zmap" if dlg.GetSelection() == 0 else "single"
    return None


def ask_layer(parent: wx.Window, default: str = "F.Cu") -> Optional[str]:
    choices = ["F.Cu", "B.Cu"]
    try_default = choices.index(default) if default in choices else 0
    with wx.SingleChoiceDialog(parent, "Copper layer:", "Import Wire Sections", choices) as dlg:
        dlg.SetSelection(try_default)
        if dlg.ShowModal() == wx.ID_OK:
            return choices[dlg.GetSelection()]
    return None


def ask_save_copies(parent: wx.Window) -> bool:
    with wx.MessageDialog(parent,
        "Save a copy of the PCB into each source folder that contained a Wire_Sections.txt?\n\n"
        "• A copy is saved *after* import finishes.\n"
        "• File name pattern: <boardname>__imported.kicad_pcb (or Imported_Board.kicad_pcb if unsaved).",
        "Save copies?", wx.YES_NO | wx.CENTRE | wx.ICON_QUESTION) as dlg:
        return dlg.ShowModal() == wx.ID_YES


# ----- Saving ---------------------------------------------------------------

def save_copy_in_folder(board: "pcbnew.BOARD", folder: str, name_suffix: Optional[str] = None):
    """Save a copy of the current board in the given folder."""
    name = board.GetFileName() or ""
    base = os.path.splitext(os.path.basename(name))[0] if name else "Imported_Board"
    if name_suffix:
        safe = name_suffix.strip().replace(os.sep, "_")
        if safe:
            base = f"{base}_{safe}"
    out_path = os.path.join(folder, f"{base}__imported.kicad_pcb")
    try:
        pcbnew.SaveBoard(out_path, board)
        return out_path
    except Exception as e:
        return f"[save failed] {folder}: {e}"


# ----- Action Plugin ---------------------------------------------------------

class ImportWireSectionsPlugin(pcbnew.ActionPlugin):
    def defaults(self):
        self.name = "Import Wire Sections → Z→Layer mapping + auto layer count (v9c)"
        self.category = "Modify PCB"
        self.description = "Read Address.txt → find Wire_Sections.txt → draw tracks per Section; map Z to layers and set copper layer count automatically."
        self.show_toolbar_button = False
        self.icon_file_name = ""

    def Run(self):
        board = pcbnew.GetBoard()
        if board is None:
            wx.MessageBox("No PCB is open. Please open your PCB first.", "Import Wire Sections", wx.OK | wx.ICON_WARNING)
            return

        parent = None

        Address_path = ask_file_open(parent, "Choose Address.txt (one folder per line)", "Text files (*.txt)|*.txt|All files|*.*")
        if not Address_path:
            return

        # Gather all Wire_Sections.txt
        try:
            folders = read_Address_file(Address_path)
        except Exception as e:
            wx.MessageBox(f"Failed to read Address.txt:\n{e}", "Import Wire Sections", wx.OK | wx.ICON_ERROR)
            return

        if not folders:
            wx.MessageBox("Address.txt contains no valid folders.", "Import Wire Sections", wx.OK | wx.ICON_WARNING)
            return

        txt_files: List[str] = []
        for folder in folders:
            txt_files.extend(find_wire_sections_in_folder(folder))

        if not txt_files:
            wx.MessageBox("No Wire_Sections.txt found in any listed folder.", "Import Wire Sections", wx.OK | wx.ICON_WARNING)
            return

        # Parameters
        mode = ask_mode(parent)
        if mode is None:
            return
        width_mm = ask_track_width(parent, default_mm=0.25)
        if width_mm is None:
            return
        fixed_layer = None
        if mode == "single":
            fixed_layer = ask_layer(parent, default="F.Cu")
            if fixed_layer is None:
                return

        save_copies = ask_save_copies(parent)

        total_segments = 0
        summary_lines: List[str] = []
        saved_paths: List[str] = []

        total_files = len(txt_files)

        for idx, txt_path in enumerate(txt_files):
            folder = os.path.dirname(txt_path)
            try:
                unit_str, _, sections = parse_wire_sections(txt_path)
            except Exception as e:
                summary_lines.append(f"❌ {os.path.basename(txt_path)} — ERROR: {e}")
                continue

            need_cleanup = idx < total_files - 1
            segs, per_layer, new_segments = draw_sections(
                board,
                sections,
                mode,
                width_mm,
                fixed_layer,
                collect_segments=need_cleanup)
            total_segments += segs
            layer_stats = ", ".join([f"{k}:{v}" for k, v in sorted(per_layer.items())]) if per_layer else "none"
            summary_lines.append(f"✓ {os.path.basename(txt_path)} — Sections:{len(sections)} → Segments:{segs} (units:{unit_str}; layers:{layer_stats})")

            if save_copies:
                folder_suffix = os.path.basename(os.path.normpath(folder)) or None
                saved = save_copy_in_folder(board, folder, folder_suffix)
                saved_paths.append(str(saved))

            if need_cleanup and new_segments:
                for seg in new_segments:
                    try:
                        board.Remove(seg)
                    except Exception:
                        pass

        msg = "Import finished.\n\n" + "\n".join(summary_lines) + f"\n\nTotal segments created: {total_segments}"
        if save_copies:
            msg += "\n\nSaved copies:\n" + "\n".join(saved_paths)

        wx.MessageBox(msg, "Import Wire Sections", wx.OK | wx.ICON_INFORMATION)


# Helper functions from earlier versions (placed below to avoid NameError)
def find_wire_sections_in_folder(folder: str) -> List[str]:
    out: List[str] = []
    direct = os.path.join(folder, "Wire_Sections.txt")
    if os.path.isfile(direct):
        out.append(direct)
        return out
    for root, _, files in os.walk(folder):
        if "Wire_Sections.txt" in files:
            out.append(os.path.join(root, "Wire_Sections.txt"))
    return out


def read_Address_file(Address_path: str) -> List[str]:
    folders: List[str] = []
    with open(Address_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if os.path.isdir(line):
                folders.append(line)
            else:
                print(f"[Address] Warning: not a folder → {line}")
    return folders


ImportWireSectionsPlugin().register()
