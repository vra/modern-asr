"""Allow running Modern ASR as a module: python -m modern_asr"""

from modern_asr.cli import main

if __name__ == "__main__":
    import sys
    sys.exit(main())
