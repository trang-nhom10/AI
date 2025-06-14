import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Danh sách nhãn
class_names = ['Benign', 'Malignant', 'Normal']

@st.cache_resource
def load_model():
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))
    model.load_state_dict(torch.load("resnet50_fineTuning.pth", map_location='cpu'))
    model.eval()
    return model

model = load_model()

st.title("🔬 Dự đoán Ung thư Vú từ Ảnh Siêu Âm (ResNet-50)")

uploaded_file = st.file_uploader("📤 Tải ảnh siêu âm", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Ảnh bạn đã tải lên", use_column_width=True)

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    input_tensor = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
        label = class_names[predicted.item()]

    st.success(f"✅ Kết quả dự đoán: **{label}**")
