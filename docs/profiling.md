# Profiling

Existing measurement methodology is retained. Preprocessing includes conversion and model-specific setup; training includes the existing validation/checkpoint-selection work. Inference is warmed, synchronized and repeated; record mean, median and population standard deviation in ms. CUDA allocator allocated/reserved peaks are distinct from total process VRAM. CPU RAM is process RSS sampled every 10ms in MiB. HardwareProfile.collect() exposes actual device/CPU/RAM/OS/software fields.
