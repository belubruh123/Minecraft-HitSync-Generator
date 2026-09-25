"""GUI package.

Python 3.13.0 on Windows has a bug where virtual environments can't locate
the Tcl/Tk runtime ("Can't find a usable init.tcl"). Point Tk at the base
installation's copy before tkinter is first used.
"""
import glob
import os
import sys


def _fix_tcl_paths():
    if sys.platform != "win32":
        return
    root = os.path.join(sys.base_prefix, "tcl")
    for env, pattern in (("TCL_LIBRARY", "tcl8*"), ("TK_LIBRARY", "tk8*")):
        if os.environ.get(env):
            continue
        for cand in sorted(glob.glob(os.path.join(root, pattern)), reverse=True):
            if os.path.isfile(os.path.join(cand, "init.tcl" if env == "TCL_LIBRARY" else "tk.tcl")):
                os.environ[env] = cand
                break


_fix_tcl_paths()
