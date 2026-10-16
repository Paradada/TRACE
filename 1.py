
import torch
import dgl
import rdkit
import numpy
import sklearn

print("torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("torch CUDA:", torch.version.cuda)
print("DGL:", dgl.__version__)
print("NumPy:", numpy.__version__)
print("scikit-learn:", sklearn.__version__)
print("RDKit:", rdkit.__version__)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
