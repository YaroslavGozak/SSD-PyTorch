"""CLI alias for the inspect phase."""
import sys
from .stages import main

if __name__ == "__main__":
    raise SystemExit(main(["inspect", *sys.argv[1:]]))
