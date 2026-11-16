"""CLI alias for the calibrate phase."""
import sys
from .stages import main

if __name__ == "__main__":
    raise SystemExit(main(["calibrate", *sys.argv[1:]]))
