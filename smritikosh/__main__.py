"""Entry point for ``python -m smritikosh``.

The ``smritikosh`` console script only works when the installing environment's
``bin`` directory is on PATH, which is not the case for ``pip install --user``
on macOS. Importing the package is enough to reach the CLI this way.
"""

from smritikosh.cli import main

if __name__ == "__main__":
    main()
