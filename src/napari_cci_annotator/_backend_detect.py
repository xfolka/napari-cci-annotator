import napari_cci_annotator._config as _config

def select_backend():
    backend = None
    # First check for TensorFlow GPU
    try:
        import torch
        if torch.cuda.is_available():
            backend = torch.cuda.get_device_name(0)
            return True, backend
    except ImportError:
        pass  # TensorFlow not installed
    
    # If no TensorFlow GPU, check OpenVINO
    try:
        from openvino.runtime import Core
        core = Core()
        devices = core.available_devices
        if 'GPU' in devices:
            backend = _config.OPENVINO_BACKEND_GPU
        elif 'CPU' in devices:
            backend = _config.OPENVINO_BACKEND_CPU
            
        return True, backend
    except ImportError:
        pass  # OpenVINO not installed
    
    # Fallback options
    
    if backend:
        return True, backend
    
    return False, "No supported backend"


def get_list_of_backends():
    backends = []
    # First check for TensorFlow GPU
    try:
        import torch
        if torch.cuda.is_available():
            backends.append(torch.cuda.get_device_name(0))
    except ImportError:
        pass  # TensorFlow not installed
    
    # If no TensorFlow GPU, check OpenVINO
    try:
        from openvino import Core
        core = Core()
        devices = core.available_devices
        if 'GPU' in devices:
            backends.append(_config.OPENVINO_BACKEND_GPU)
        if 'CPU' in devices:
            backends.append(_config.OPENVINO_BACKEND_CPU)
            
    except ImportError:
        pass  # OpenVINO not installed
    
    return backends