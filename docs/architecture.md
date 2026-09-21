# Architecture

Python Benchmark and CLI matrix share benchmark/matrix.py scheduling and benchmark/runner.py execution. Models stay unchanged. External workers live in the installed package and execute in isolated interpreters through versioned JSON and numeric NPZ. Results retain canonical identities and physical hardware fingerprints. Separate dataset loaders, model metadata, execution and result layers leave room for future tasks without claiming those tasks are implemented.
