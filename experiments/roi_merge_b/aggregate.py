"""CLI alias for the aggregate phase."""
import sys
from .stages import main

if __name__ == "__main__":
    raise SystemExit(main(["aggregate", *sys.argv[1:]]))
