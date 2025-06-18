# app.py
import streamlit as st
import matplotlib.pyplot as plt
import torch
import numpy as np
from ntgenerator import DepthwiseNuisanceGenerator

"""
Streamlit web app that visualizes the distance between noisy and cleaned examples. Allows users to change transformation
intensity per examples, with visualizations.
"""


INPUT_PATH = "data/random-windows/cartpole_nuisance.npz"
GEN_PATH = "model/generator/nuisance_transformations.pth"
OUTPUT_PATH = "data/random-windows/cartpole_cleaned_by_generator.npz"
BATCH_SIZE = 1024
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load model
@st.cache_resource
def load_generator():
    generator = DepthwiseNuisanceGenerator().to(DEVICE)
    checkpoint = torch.load(GEN_PATH, map_location=DEVICE)
    generator.load_state_dict(checkpoint['state_dict'])
    generator.eval()
    return generator

g = load_generator()

# Load data
data = np.load("data/random-windows/cartpole_nuisance.npz")
x_noisy = torch.tensor(data['x'], dtype=torch.float32)
y = data['y']

# UI Controls
st.title("Nuisance Inspector")
idx = st.slider("Choose sample index", 0, len(x_noisy)-1, 0)
dim_to_edit = st.selectbox("Feature to adjust", ["Cart Pos", "Cart Vel", "Pole Angle", "Pole Ang Vel"], index=2)
intensity = st.slider("Residual intensity (0 = suppress, 1 = full)", 0.0, 1.0, 1.0, 0.05)

# Process
x = x_noisy[idx:idx+1]          # shape (1, 50, 4)
with torch.no_grad():
    g_x = g(x)                  # shape (1, 50, 4)

residual = g_x - x
dim_idx = ["Cart Pos", "Cart Vel", "Pole Angle", "Pole Ang Vel"].index(dim_to_edit)

# Adjusted transformation
residual[:, :, dim_idx] *= intensity
x_cleaned = x + residual        # override this feature only

# Plotting
fig, ax = plt.subplots(4, 1, figsize=(10, 8))
feature_names = ["Cart Pos", "Cart Vel", "Pole Angle", "Pole Ang Vel"]

for d in range(4):
    ax[d].plot(x[0,:,d], label='Noisy', color='orange')
    ax[d].plot(g_x[0,:,d], label='Auto Cleaned', color='green')
    ax[d].plot(x_cleaned[0,:,d], label='User Cleaned', color='blue')
    ax[d].set_title(feature_names[d])
    ax[d].legend()

st.pyplot(fig)

# Optionally save feedback
if st.button("Save this correction"):
    corrected_np = x_cleaned[0].numpy()
    np.save(f"user_correction_{idx}.npy", corrected_np)
    st.success("✅ Correction saved!")
