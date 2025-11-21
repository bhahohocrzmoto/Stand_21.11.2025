import os
import sys
import time
import win32com.client


def run_fasthenry(inp_path, options=""):
    # Make path absolute and ensure it exists
    inp_path = os.path.abspath(inp_path)
    if not os.path.isfile(inp_path):
        raise FileNotFoundError(f"Input file not found: {inp_path}")

    # Build command line string as FastHenry2 expects:
    #   "full\path\file.inp"
    # path must be quoted because of spaces
    if options:
        cmdline = f'"{inp_path}" {options}'
    else:
        cmdline = f'"{inp_path}"'

    print(f"Calling FastHenry2 with:\n  {cmdline}")

    # Create COM object (FastHenry2.Document)
    fh = win32com.client.Dispatch("FastHenry2.Document")

    # Start simulation – IGNORE the return value (can be False/None)
    _ = fh.Run(cmdline)

    # Poll until FastHenry2 is done.
    # In the VB/Matlab examples IsRunning is a method, but here it behaves
    # like a property, so we don't use parentheses.
    max_wait_s = 600  # safety timeout: 10 minutes
    t0 = time.time()

    while True:
        try:
            running = bool(fh.IsRunning)
        except Exception as e:
            print(f"Error reading IsRunning: {e}")
            break

        if not running:
            break

        if time.time() - t0 > max_wait_s:
            print("Timeout: FastHenry2 still running after 10 minutes.")
            break

        time.sleep(0.5)

    # At this point Zc.mat should be written next to the .inp file
    zc_path = os.path.join(os.path.dirname(inp_path), "Zc.mat")
    return zc_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_fasthenry.py path\\to\\file.inp [extra FastHenry options]")
        sys.exit(1)

    inp = sys.argv[1]
    # Any extra arguments after the inp are passed to FastHenry2 as command line options
    extra_opts = " ".join(sys.argv[2:])  # e.g. "-r2 -M"

    zc_file = run_fasthenry(inp, extra_opts)

    if os.path.isfile(zc_file):
        print(f"FastHenry2 finished. Zc.mat saved at:\n  {zc_file}")
    else:
        print("FastHenry2 finished but Zc.mat was not found where expected:")
        print(f"  {zc_file}")
