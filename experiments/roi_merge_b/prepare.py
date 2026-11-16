"""CLI alias for the prepare phase."""
import sys
from .stages import main

if __name__ == "__main__":
    raise SystemExit(main(["prepare", *sys.argv[1:]]))
