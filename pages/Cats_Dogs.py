import streamlit as st
import torch
import io
from torchvision import transforms as T
from PIL import Image

st.title('Сейчас узнаем кот или пес на твоем фото!')


# Загружаем модель и веса
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = torch.load('pages/model_fine.pt', map_location=torch.device(device))
model.load_state_dict(torch.load('pages/model_fine_state_dict.pt', map_location=torch.device(device)))
model.eval()

def get_prediction(img):
  
    class_dict = {
        0 : 'Кот',
        1 : 'Пес'
    }
    
    preproccessing = T.Compose([
        T.Resize((244, 244)),
        T.ToTensor()
    ])
    img_pr = preproccessing(img)
    cls_pred = model(img_pr.to(device).unsqueeze(0)).sigmoid().item()

    return class_dict[round(cls_pred)], cls_pred


def load_image():
    uploaded_file = st.file_uploader('Живо загружай фото!', type=['jpg', 'jpeg'])
    if uploaded_file is not None:
        img_data = uploaded_file.getvalue()
        st.image(img_data)
        img = Image.open(io.BytesIO(img_data))
        return img


img = load_image()


result = st.button('Ну давай посмотрим кто тут у нас...')
if result:
    pet, prob = get_prediction(img)
    if round(prob) == 1:
        st.markdown(f'**Братик, это {pet}:dog: c вероятностью {prob:.3f}!**')
    else:
        st.markdown(f'**Братик, это {pet}:cat: c вероятностью {1 - prob:.3f}!**')