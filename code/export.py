import torch
from model_builder import YoloV1_model  # Replace with your actual model class or import

# 1. Initialize your model architecture
model = YoloV1_model()

# 2. Load the .pth state dictionary or full checkpoint
model.load_state_dict(torch.load("../model/Epoch-86.pth", map_location=torch.device("cpu"))['model_state_dict'])
model.eval()  # Set the model to evaluation mode

# 3. Create a dummy input tensor matching your model's expected input shape
example_input = torch.rand(1, 3, 446, 446)  # Adjust dimensions to match your model

# 4. Trace the model
traced_model = torch.jit.trace(model, example_input)

# 5. Save as a .pt file
traced_model.save("model.pt")