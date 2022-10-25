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

    result1_msg = st.empty()
    result2_msg = st.empty()
    st.markdown("-------------------------------------------------------------------------------")

    def predict_class(img) :
        classifier_model = tf.keras.models.load_model(r'potatoes.h5', compile = False)
        print('Model Loaded Succafully !!')
        image = Image.open(img).convert("L")
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
        imgg = st.file_uploader(label="Choose a picture : ", type=['jpeg', 'jpg', 'png'], key="xray")

        st.markdown("<h3 style='text-align: center; color: white;'>  </h3>", unsafe_allow_html=True)

        picture = st.camera_input("Take a picture : ")
        if picture is not None:
            showed_img=Image.open(picture)
            showed_img= showed_img.resize((256, 256))
            imageLocation.image(showed_img)

            result1= pneumoniapredictPage(picture)
            with result1_msg.container():
                if result1 == "Normal" : 
                    st.success("Your Lungs are Healthy")
                elif result1 == "penumonia":
                    st.error("There is Pneumonia, You Should Go to The Doctor")
        
        if imgg is not None:
            #show to ui
            showed_img=Image.open(imgg)
            showed_img= showed_img.resize((256, 256))
            imageLocation.image(showed_img)
            # for prediction
            result2= pneumoniapredictPage(imgg)
            with result2_msg.container():
                if result2 == "Normal" : 
                    st.success("Your Lungs are Healthy")
                elif result2 == "penumonia":
                    st.error("There is Pneumonia, You Should Go to The Doctor")
    except:
        st.markdown("<h3 style='text-align: center; color: white;'>Try Another Pictuer !</h3>", unsafe_allow_html=True)



# --------------------------------------------------ABOUT PAGE-----------------------------------------------------------
if selected =='About':
    hide_streamlit_style = """
                <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                </style>
                """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
    st.markdown("<h1 style='text-align: center; color: white;'>How I Am ?</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: white;'>My Name is Abdelhamid Adel</h3>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: white;'>Data Scientist | Data analyst l knowledgeable in Machine learning - Deep learning - NLP - Computer Vision</h3>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: white;'> I am From Cairo, Egypt</h3>", unsafe_allow_html=True)

    button_styl="""<style>
    button {
  background-color: #c2fbd7;
  border-radius: 100px;
  box-shadow: rgba(44, 187, 99, .2) 0 -25px 18px -14px inset,rgba(44, 187, 99, .15) 0 1px 2px,rgba(44, 187, 99, .15) 0 2px 4px,rgba(44, 187, 99, .15) 0 4px 8px,rgba(44, 187, 99, .15) 0 8px 16px,rgba(44, 187, 99, .15) 0 16px 32px;
  color: green;
  cursor: pointer;
  display: inline-block;
  font-family: CerebriSans-Regular,-apple-system,system-ui,Roboto,sans-serif;
  padding: 7px 20px;
  text-align: center;
  text-decoration: none;
  transition: all 250ms;
  border: 0;
  font-size: 16px;
  user-select: none;
  -webkit-user-select: none;
  touch-action: manipulation;
  vertical-align: middle;
  margin: 0;
  position: absolute;
  top: 50%;
  left: 50%;
  -ms-transform: translate(-50%, -50%);
}

button:hover {
  box-shadow: rgba(44,187,99,.35) 0 -25px 18px -14px inset,rgba(44,187,99,.25) 0 1px 2px,rgba(44,187,99,.25) 0 2px 4px,rgba(44,187,99,.25) 0 4px 8px,rgba(44,187,99,.25) 0 8px 16px,rgba(44,187,99,.25) 0 16px 32px;
  transform: scale(1.05) rotate(-1deg);
}
    </style>"""

    st.markdown(button_styl, unsafe_allow_html=True) 


    st.write(f'''
    <a target="_blank" href="https://github.com/AbdelhamidADel">
        <button>
            My GitHub
        </button>
    </a>
    ''',
    unsafe_allow_html=True)
