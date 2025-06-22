import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io

# Page configuration
st.set_page_config(
    page_title="MNIST Digit Generator",
    page_icon="🔢",
    layout="wide"
)

# ---------------------- Model Definition ----------------------

class ConditionalGenerator(nn.Module):
    def __init__(self, in_features, out_features, num_classes=10):
        super(ConditionalGenerator, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_classes = num_classes

        self.fc1 = nn.Linear(in_features + num_classes, 32)
        self.relu1 = nn.LeakyReLU(0.2)
        self.fc2 = nn.Linear(32, 64)
        self.relu2 = nn.LeakyReLU(0.2)
        self.fc3 = nn.Linear(64, 128)
        self.relu3 = nn.LeakyReLU(0.2)
        self.fc4 = nn.Linear(128, out_features)
        self.dropout = nn.Dropout(0.3)
        self.tanh = nn.Tanh()

    def forward(self, x, labels):
        batch_size = x.shape[0]
        labels_onehot = torch.zeros(batch_size, self.num_classes).to(x.device)
        labels_onehot.scatter_(1, labels.unsqueeze(1), 1)

        x = torch.cat([x, labels_onehot], dim=1)
        x = self.dropout(self.relu1(self.fc1(x)))
        x = self.dropout(self.relu2(self.fc2(x)))
        x = self.dropout(self.relu3(self.fc3(x)))
        x = self.tanh(self.fc4(x))
        return x

# ---------------------- Load Model ----------------------

@st.cache_resource
def load_generator():
    import pickle as pkl
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        with open('generator_model.pkl', 'rb') as f:
            generator = pkl.load(f)
        generator.to(device)
        generator.eval()
        st.success("✅ Model loaded successfully!")
        return generator, device

    except FileNotFoundError:
        st.error("❌ Model file not found! Please train and save 'generator_model.pkl'")
        return None, device

    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None, device

# ---------------------- Image Generation ----------------------

def generate_images_for_digit(generator, digit, num_images=5, z_size=100, device='cpu'):
    if generator is None:
        return None

    generator.eval()
    with torch.no_grad():
        z = np.random.uniform(-1, 1, size=(num_images, z_size))
        z = torch.from_numpy(z).float().to(device)

        labels = torch.full((num_images,), digit, dtype=torch.long).to(device)
        fake_images = generator(z, labels)
        fake_images = (fake_images + 1) / 2
        fake_images = fake_images.view(num_images, 1, 28, 28)

    return fake_images.cpu()

def tensor_to_pil(tensor_image):
    img_array = tensor_image.squeeze().numpy()
    img_array = (img_array * 255).astype(np.uint8)
    return Image.fromarray(img_array, mode='L')

# ---------------------- Streamlit App ----------------------

def main():
    st.title("🔢 MNIST Digit Generator")
    st.markdown("Generate handwritten digits using a trained Conditional GAN!")

    generator, device = load_generator()
    if generator is None:
        st.stop()

    # Sidebar: Controls
    st.sidebar.header("🛠️ Generation Settings")

    digit = st.sidebar.selectbox(
        "Select Digit to Generate",
        options=list(range(10)),
        index=4,
        help="Choose which digit (0-9) you want to generate"
    )

    num_images = st.sidebar.slider(
        "Number of Images",
        min_value=1,
        max_value=10,
        value=5,
        help="How many images to generate"
    )

    # Button to trigger generation
    if st.sidebar.button("🎯 Generate Images", type="primary"):
        with st.spinner(f"Generating {num_images} images of digit {digit}..."):
            generated_images = generate_images_for_digit(
                generator, digit, num_images, device=device
            )

            if generated_images is not None:
                st.success(f"✨ Generated {num_images} images of digit {digit}!")
                st.subheader(f"Generated Digit: {digit}")

                cols = st.columns(min(num_images, 5))
                for i in range(num_images):
                    col_idx = i % 5
                    with cols[col_idx]:
                        pil_image = tensor_to_pil(generated_images[i])
                        pil_image = pil_image.resize((112, 112), Image.Resampling.NEAREST)
                        st.image(pil_image, caption=f"Image {i+1}", use_column_width=True)

                    if (i + 1) % 5 == 0 and i + 1 < num_images:
                        cols = st.columns(min(num_images - i - 1, 5))
            else:
                st.error("Failed to generate images. Please check your model.")

    # Sidebar: Info
    st.sidebar.markdown("---")
    st.sidebar.info(
        "ℹ️ **How it works:**\n\n"
        "This app uses a Conditional GAN trained on the MNIST dataset. "
        "Select a digit and click generate to create new handwritten digit images!"
    )

    st.sidebar.markdown("---")
    device_info = "🖥️ **Device:** " + ("GPU" if device.type == "cuda" else "CPU")
    st.sidebar.markdown(device_info)

if __name__ == "__main__":
    main()
