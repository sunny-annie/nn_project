import torch
import json
import urllib.request, json
import streamlit as st
from torchvision.models import inception_v3, Inception_V3_Weights
from torchvision import transforms as T
from torchvision import io
from PIL import Image



with urllib.request.urlopen("https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json") as url:
    labels = json.load(url)
    
decode = lambda x: labels[str(x)][1]

model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
model.eval()

uploaded_file = st.file_uploader("Выберите изображение в формате jpeg или jpg...", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Загруженное изображение', use_column_width=True)

    # Обработка изображения
    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])
    image_tensor = transform(image).unsqueeze(0)

    # Предсказание класса
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        top_probabilities, top_classes = torch.topk(probabilities, k=3)
        top_classes = top_classes.tolist()

        

    st.write('Топ-3 предсказания:')
    for i in range(3):
        st.write(f'{decode(top_classes[i])}: {top_probabilities[i]:.2f}')