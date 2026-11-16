"""CLI alias for the evaluate phase."""
import sys
from .stages import main

if __name__ == "__main__":
    raise SystemExit(main(["evaluate", *sys.argv[1:]]))
