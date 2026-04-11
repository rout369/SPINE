"""
MNIST Loader for YOUR files (t10k-images.idx3-ubyte, etc.)
"""

import numpy as np
import struct
import os

class MNISTLoader:
    def __init__(self, data_path="./mnist_data"):
        self.data_path = data_path
    
    def load_images(self, filename):
        """Load MNIST images from raw file"""
        filepath = os.path.join(self.data_path, filename)
        with open(filepath, 'rb') as f:
            # Read header (magic number, number of images, rows, cols)
            magic = struct.unpack('>I', f.read(4))[0]
            num_images = struct.unpack('>I', f.read(4))[0]
            rows = struct.unpack('>I', f.read(4))[0]
            cols = struct.unpack('>I', f.read(4))[0]
            
            # Read image data
            num_pixels = rows * cols
            images = np.frombuffer(f.read(num_images * num_pixels), dtype=np.uint8)
            images = images.reshape(num_images, rows, cols)
            
        return images.astype(np.float32) / 255.0
    
    def load_labels(self, filename):
        """Load MNIST labels from raw file"""
        filepath = os.path.join(self.data_path, filename)
        with open(filepath, 'rb') as f:
            magic = struct.unpack('>I', f.read(4))[0]
            num_labels = struct.unpack('>I', f.read(4))[0]
            labels = np.frombuffer(f.read(num_labels), dtype=np.uint8)
        return labels
    
    def get_train(self):
        """Return training images and labels"""
        # Use YOUR exact filenames (with dots, not hyphens)
        images = self.load_images('train-images.idx3-ubyte')
        labels = self.load_labels('train-labels.idx1-ubyte')
        return images, labels
    
    def get_test(self):
        """Return test images and labels"""
        images = self.load_images('t10k-images.idx3-ubyte')
        labels = self.load_labels('t10k-labels.idx1-ubyte')
        return images, labels


# Test it
if __name__ == "__main__":
    loader = MNISTLoader()
    
    print("Loading MNIST from your files...")
    print("-" * 40)
    
    # Load training data
    train_images, train_labels = loader.get_train()
    print(f"✅ Training images: {train_images.shape}")
    print(f"✅ Training labels: {train_labels.shape}")
    print(f"   First label: {train_labels[0]}")
    
    # Load test data
    test_images, test_labels = loader.get_test()
    print(f"\n✅ Test images: {test_images.shape}")
    print(f"✅ Test labels: {test_labels.shape}")
    print(f"   First label: {test_labels[0]}")
    
    # Verify values
    print(f"\n📊 Image pixel range: [{train_images.min():.2f}, {train_images.max():.2f}]")
    print(f"📊 Image data type: {train_images.dtype}")
    
    print("\n🎉 MNIST loaded successfully!")



# """
# MNIST Data Loader for SpiNNaker Framework
# Converts MNIST images to spike trains or raw tensors
# """

# import numpy as np
# import struct
# import gzip
# import urllib.request
# import os

# class MNISTLoader:
#     """Load MNIST dataset and convert to your tensor format"""
    
#     def __init__(self, data_path="./mnist_data"):
#         self.data_path = data_path
#         os.makedirs(data_path, exist_ok=True)
#         self.download_if_needed()
    
#     def download_if_needed(self):
#         """Download MNIST if not already present"""
#         files = {
#             'train_images': 'train-images-idx3-ubyte.gz',
#             'train_labels': 'train-labels-idx1-ubyte.gz',
#             'test_images': 't10k-images-idx3-ubyte.gz',
#             'test_labels': 't10k-labels-idx1-ubyte.gz'
#         }
        
#         base_url = 'http://yann.lecun.com/exdb/mnist/'
        
#         for name, filename in files.items():
#             filepath = os.path.join(self.data_path, filename)
#             if not os.path.exists(filepath):
#                 print(f"Downloading {filename}...")
#                 urllib.request.urlretrieve(base_url + filename, filepath)
    
#     def load_images(self, filename):
#         """Load MNIST images"""
#         filepath = os.path.join(self.data_path, filename)
#         with gzip.open(filepath, 'rb') as f:
#             # Read header
#             magic = struct.unpack('>I', f.read(4))[0]
#             num_images = struct.unpack('>I', f.read(4))[0]
#             rows = struct.unpack('>I', f.read(4))[0]
#             cols = struct.unpack('>I', f.read(4))[0]
            
#             # Read image data
#             num_pixels = rows * cols
#             images = np.frombuffer(f.read(num_images * num_pixels), dtype=np.uint8)
#             images = images.reshape(num_images, rows, cols)
            
#         return images.astype(np.float32) / 255.0  # Normalize to [0, 1]
    
#     def load_labels(self, filename):
#         """Load MNIST labels"""
#         filepath = os.path.join(self.data_path, filename)
#         with gzip.open(filepath, 'rb') as f:
#             magic = struct.unpack('>I', f.read(4))[0]
#             num_labels = struct.unpack('>I', f.read(4))[0]
#             labels = np.frombuffer(f.read(num_labels), dtype=np.uint8)
#         return labels
    
#     def get_train(self):
#         """Return training images and labels"""
#         images = self.load_images('train-images-idx3-ubyte.gz')
#         labels = self.load_labels('train-labels-idx1-ubyte.gz')
#         return images, labels
    
#     def get_test(self):
#         """Return test images and labels"""
#         images = self.load_images('t10k-images-idx3-ubyte.gz')
#         labels = self.load_labels('t10k-labels-idx1-ubyte.gz')
#         return images, labels
    
#     def to_spikes_rate(self, images, time_steps=100):
#         """
#         Convert images to rate-coded spike trains
#         Higher pixel intensity = more spikes
#         """
#         batch_size = images.shape[0]
#         pixels = images.shape[1] * images.shape[2]
        
#         # Reshape images to [batch, pixels]
#         images_flat = images.reshape(batch_size, pixels)
        
#         # Poisson spike generation
#         spike_train = np.random.random((time_steps, batch_size, pixels))
#         spikes = (spike_train < images_flat).astype(np.float32)
        
#         return spikes
    
#     def to_spikes_latency(self, images, max_time=100):
#         """
#         Convert images to latency-coded spikes
#         Higher intensity = earlier spike
#         """
#         batch_size = images.shape[0]
#         pixels = images.shape[1] * images.shape[2]
#         images_flat = images.reshape(batch_size, pixels)
        
#         # Convert intensity to spike time (inverse relationship)
#         spike_times = (1 - images_flat) * (max_time - 1)
        
#         # Create spike train
#         spike_train = np.zeros((max_time, batch_size, pixels))
#         for t in range(max_time):
#             spike_train[t] = (spike_times == t).astype(np.float32)
        
#         return spike_train