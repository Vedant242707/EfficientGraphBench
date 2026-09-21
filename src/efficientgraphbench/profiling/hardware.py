"""Report the actual PyTorch execution backend."""

import torch


def hardware_info():
    info = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "device_name": "CPU",
    }
    if info["cuda_available"]:
        properties = torch.cuda.get_device_properties(0)
        free, total = torch.cuda.mem_get_info(0)
        info.update(
            device_name=properties.name,
            gpu_memory_mb=total / 1024**2,
            gpu_free_memory_mb=free / 1024**2,
        )
    return info


class HardwareProfile:
    """Actual machine properties; memory values are MiB."""

    def __init__(self, record):
        self.record = record

    @classmethod
    def collect(cls):
        from efficientgraphbench.benchmark.identity import physical_hardware

        software = hardware_info()
        physical, fingerprint = physical_hardware("cuda" if software["cuda_available"] else "cpu")
        return cls({**physical, **software, "hardware_fingerprint": fingerprint})

    def to_dict(self):
        return dict(self.record)

    def __getattr__(self, name):
        if name in self.record:
            return self.record[name]
        raise AttributeError(name)
