import tensorflow as tf
import torch

print("GPU available:", tf.config.list_physical_devices('GPU'))
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
print("GPU name:", torch.cuda.get_device_name(0))
print(torch.cuda.get_device_properties(0))