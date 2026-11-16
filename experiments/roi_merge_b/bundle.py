"""CLI alias for the bundle phase."""
import sys
from .stages import main

if __name__ == "__main__":
    raise SystemExit(main(["bundle", *sys.argv[1:]]))
