import torch.nn as nn

class MyCustomClass(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MyCustomClass, self).__init__()
        # ... your class implementation ... 

# Optional: Export the class for easy import
__all__ = ['MyCustomClass'] 