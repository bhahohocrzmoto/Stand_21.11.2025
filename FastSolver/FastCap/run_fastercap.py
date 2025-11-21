import os
import sys
import time
import win32com.client


def run_fastercap(inp_path, options=None, eps_r=1.0):
    """
    Run FasterCap on a single geometry file (.txt) and save the
    Maxwell capacitance matrix as a text file next to the input.

    Parameters
    ----------
    inp_path : str
        Path to FasterCap geometry file (e.g. .txt, NOT list file).
    options : str or None
        Extra command-line options, e.g. "-a0.01" for 1% rel. accuracy.
        If None or empty, defaults to "-a0.01".
        NOTE: These are passed directly to FasterCap.Run("<file> <options>").
    eps_r : float
        Homogeneous relative permittivity to apply AFTER the simulation.
        eps_r = 1.0 -> vacuum/air (no scaling).
        eps_r = 3.5 -> scale all capacitances by 3.5 (FR-4-ish).
    """
    inp_path = os.path.abspath(inp_path)
    if not os.path.isfile(inp_path):
        raise FileNotFoundError(f"Input file not found: {inp_path}")

    # Default to Auto with max error 1% if nothing is specified
    if options is None or not options.strip():
        options = "-a0.01"
    else:
        options = options.strip()

    # File is a geometry file, not a list file, so NO -l flag.
    cmdline = f'"{inp_path}" {options}'.strip()

    print("Calling FasterCap with:")
    print(f"  {cmdline}")
    print(f"  (homogeneous eps_r scaling after run: {eps_r})")

    # Create COM object for FasterCap
    FasterCap = win32com.client.Dispatch("FasterCap.Document")

    # Start simulation (ignore return value)
    FasterCap.Run(cmdline)

    # Wait until it finishes – for FasterCap, IsRunning is a method
    start = time.time()
    max_wait_s = 600  # 10 minutes safety

    while True:
        running = FasterCap.IsRunning()
        if not running:
            break

        if time.time() - start > max_wait_s:
            print("Timeout: FasterCap still running after 10 minutes.")
            break

        time.sleep(0.5)

    # Get the capacitance matrix from Automation
    cap = FasterCap.getCapacitance()

    # Apply homogeneous dielectric scaling if requested
    # (for pure-vacuum input geometries this is equivalent to solving
    #  the same structure embedded in a homogeneous eps_r medium)
    if eps_r != 1.0:
        scaled_cap = []
        for row in cap:
            scaled_cap.append([eps_r * float(val) for val in row])
        cap = scaled_cap

    # Save as plain text next to input file
    out_path = os.path.join(os.path.dirname(inp_path), "CapacitanceMatrix.txt")

    with open(out_path, "w", encoding="utf-8") as f:
        for row in cap:
            f.write(" ".join(f"{val:.6e}" for val in row) + "\n")

    print("Capacitance matrix written to:")
    print(f"  {out_path}")
    return out_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python run_fastercap.py path\\to\\geom.txt [eps_r] [extra FasterCap options]")
        print()
        print("Examples:")
        print("  # Vacuum / air (eps_r = 1), 1% auto accuracy")
        print("  python run_fastercap.py mygeom.txt")
        print()
        print("  # FR-4-ish homogeneous medium with eps_r = 3.5, 1% auto accuracy")
        print("  python run_fastercap.py mygeom.txt 3.5")
        print()
        print("  # eps_r = 2.5 and custom FasterCap options")
        print("  python run_fastercap.py mygeom.txt 2.5 -a0.005 -m1")
        sys.exit(1)

    inp = sys.argv[1]

    # Default values
    eps_r = 1.0
    extra_opts = ""

    if len(sys.argv) >= 3:
        # If the second argument looks like a float, treat it as eps_r
        # and pass the rest as FasterCap options.
        try:
            eps_r = float(sys.argv[2])
            extra_opts = " ".join(sys.argv[3:])
        except ValueError:
            # No eps_r provided, all remaining args are FasterCap options
            extra_opts = " ".join(sys.argv[2:])

    run_fastercap(inp, extra_opts, eps_r)
