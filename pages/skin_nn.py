
import streamlit as st
import torch
import io
from torchvision import transforms as T
from torchvision.models import mobilenet_v3_small
from PIL import Image

st.title("Не является рекомендацией врача!")


device = 'cuda' if torch.cuda.is_available() else 'cpu'


model = mobilenet_v3_small()
model.classifier[3] = torch.nn.Linear(1024, 1)
model.load_state_dict(torch.load('pages/skin_nn.pt'))
model.to(device)

uploaded_file = st.file_uploader('Пожалуйста, загрузите фотографию в формате .jpeg или .jpg', type=['jpg', 'jpeg'])
if uploaded_file is not None:

    if uploaded_file is not None:
        img_data = uploaded_file.getvalue()
        st.image(img_data)
        img = Image.open(io.BytesIO(img_data))

        def get_prediction(img):
            
            preproccessing = T.Compose([
                T.Resize((244, 244)),
                T.ToTensor()
            ])
            img_pr = preproccessing(img)
            cls_pred = model(img_pr.to(device).unsqueeze(0)).sigmoid().item()

            return cls_pred


        prob = get_prediction(img)
        if round(prob) != 1:
            st.write(f'Образование является доброкачественным с вероятностью {1 - prob:.3f}')
        else:
            st.write(f'Образование является злокачественным c вероятностью {prob:.3f}')






