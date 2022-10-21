import tensorflow as tf
from PIL import Image
import numpy as np
import streamlit as st
from streamlit_option_menu import option_menu
import base64

st.set_page_config(layout="centered",page_icon="🥔",page_title="Potato Disease")


with st.sidebar:
    selected = option_menu("Main Menu", ["Prediction", 'About'], 
        icons=['house', 'gear'], menu_icon="cast", default_index=1)
#----------------------------------------------------------------
def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)
set_background('back_1.jpg')
#------------------------------------------------------------------ 
if selected =='Prediction':
    hide_streamlit_style = """
                <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                </style>
                """
    img_style="""<style>
    img {
    margin-left: 150px;
    display:relative;
    width: 100%;}
    </style>"""

    st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
    st.markdown(img_style, unsafe_allow_html=True) 
    st.markdown("<h1 style='text-align: center; color: white;'>Potato Disease Classification </h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: white;'>-Model made by CNN-</h3>", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align: center; color: white;'>-Model Accuracy-99 % || Model Evaluation-98 %-</h6>", unsafe_allow_html=True)

    st.markdown("-------------------------------------------------------------------------------")

    def predict_class(img) :
        classifier_model = tf.keras.models.load_model(r'potatoes.h5', compile = False)
        print('Model Loaded Succafully !!')
        image = Image.open(img)
        shape = ((256,256,3))

        test_image = image.resize((256, 256))
        test_image = tf.keras.utils.img_to_array(test_image)
        test_image /= 255.0
        test_image = np.expand_dims(test_image, axis = 0)

        class_name = ['Potato__Early blight', 'Potato__Late blight', 'Potato__healthy']
        prediction = classifier_model.predict(test_image)
        confidence = round(100 * (np.max(prediction[0])), 2)
        final_pred = class_name[np.argmax(prediction)]
        return final_pred, confidence

    try:
        imageLocation = st.empty()
        imageLocation.image('Default_Image_Thumbnail.png')
        img = st.file_uploader(label="Potato Disease Classification : ", type=['jpeg', 'jpg', 'png'], key="xray")
        if img is not None:
            showed_img=Image.open(img)
            showed_img= showed_img.resize((256, 256))
            imageLocation.image(showed_img)
            loading_msg = st.empty()
            loading_msg.text("Predicting...")
            result, confidence = predict_class(img)
            if result =="Potato__healthy":
                st.success('Prediction : {}'.format(result))
            elif result =="Potato__Early blight":
                st.warning('Prediction : {}'.format(result))
            else:
                st.error('Prediction : {}'.format(result))
            st.write('Confidence : {}%'.format(confidence))
    except:
        st.markdown("<h3 style='text-align: center; color: white;'>Try Another Pictuer !</h3>", unsafe_allow_html=True)
if selected =='About':
    hide_streamlit_style = """
                <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                </style>
                """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
    st.header("How I Am ?")
    st.text('My Name is Abdelhamid Adel, Data scientist')
    st.text('I am From Cairo, Egypt')
    st.write(f'''
    <a target="_blank" href="https://github.com/AbdelhamidADel">
        <button>
            My GitHub
        </button>
    </a>
    ''',
    unsafe_allow_html=True)