Install CUDA toolkit (Libux/Debian)
```
apt install nvidia-cuda-toolkit
```

Also may help: nvidia-cuda-dev, nvidia-cuda-gdb


Get installed CUDA version
```
nvcc --version
```

Get CUDA capabilities supported by the hardware
```
nvidia-smi
```

Install cupy wheel (package name suffix depends on the CUDA version):
```
pip install cupy-cuda11x
```

Ensure the CUDA driver can be actually used
```python
import cupy
cupy.cuda.get_local_runtime_version()
cupy.cuda.device.Device().compute_capability
```

Reinstallation or library changes may cause CUDA to load. For example you may get an "Unknown Error". 
In that case cleaning compiled kernel cache may help:
```
rm -r "$HOME/.nv"
```

