import os
import io
import base64
import pickle
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import plotly.express as px
from streamlit_option_menu import option_menu
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="Water Prediction",
    layout="wide",
    page_icon="🌊"
)

# Getting the working directory of the main.py
working_dir = os.path.dirname(os.path.abspath(__file__))

# Load the saved models
# pickle    
def load_model():
    with open('saved_models/RandomF_model.pkl', 'rb') as f:
        return pickle.load(f)

water_model = load_model()

# Sidebar navigation
with st.sidebar:
    selected = option_menu(
        'Water Quality Prediction',
        [
            'Home',
            'Data Description',
            'Analytics',
            'Water Prediction',
            'About Us'
        ],
        menu_icon='water',
        icons=['house', 'clipboard-data', 'graph-up', 'droplet', 'people'],
        default_index=0
    )

# Load Dataset
@st.cache_data
# Dataset Describe
def load_data():
  dataset_path = os.path.join(working_dir, "dataset1/dataset_baruku.csv")
  data = pd.read_csv(dataset_path)
  return data

# Dataset Train Mix (categorical and numeric)
def load_data2():
  dataset_path2 = os.path.join(working_dir, "dataset1/dataset_desc.csv")
  data = pd.read_csv(dataset_path2)
  return data

# All Numeric Dataset Train
def load_data3():
  dataset_path3 = os.path.join(working_dir, "dataset1/df_baru.csv")
  data = pd.read_csv(dataset_path3)
  return data

# Home Page
if selected == 'Home':
    
  # Load the image background
  image_path = os.path.join(working_dir, "Foto/background.png")
  image = Image.open(image_path)
  buffered = io.BytesIO()
  image.save(buffered, format="PNG")
  img_str = base64.b64encode(buffered.getvalue()).decode()

  # Load the image logo merdeka
  image_path = os.path.join(working_dir, "Foto/merdeka.png")
  image = Image.open(image_path)
  buffered = io.BytesIO()
  image.save(buffered, format="PNG")
  img_str1 = base64.b64encode(buffered.getvalue()).decode()

  # Load the image logo msib
  image_path = os.path.join(working_dir, "Foto/msib.png")
  image = Image.open(image_path)
  buffered = io.BytesIO()
  image.save(buffered, format="PNG")
  img_str2 = base64.b64encode(buffered.getvalue()).decode()

    # Load the image logo GreatEdu
  image_path = os.path.join(working_dir, "Foto/greatedu.png")
  image = Image.open(image_path)
  buffered = io.BytesIO()
  image.save(buffered, format="PNG")
  img_str6 = base64.b64encode(buffered.getvalue()).decode()

  # Load the image logo Tut Wuri Handayani
  image_path = os.path.join(working_dir, "Foto/Tut Wuri Handayani.png")
  image = Image.open(image_path)
  buffered = io.BytesIO()
  image.save(buffered, format="PNG")
  img_str3 = base64.b64encode(buffered.getvalue()).decode()

  # Load the image logo air
  image_path = os.path.join(working_dir, "Foto/air.jpeg")
  image = Image.open(image_path)
  buffered = io.BytesIO()
  image.save(buffered, format="JPEG")
  img_str4 = base64.b64encode(buffered.getvalue()).decode()

  st.markdown(
      f"""
      <style>
      .stApp {{
          background-image: url("data:image/jpeg;base64,{img_str}");
          background-size: cover;
          background-position: center;
          background-repeat: no-repeat;
      }}
      .header {{
          display: flex;
          justify-content: space-between;
      }}
      .kumlogo {{
          display: flex;
          background-position: center;
          margin-top: 20px;
      }}
      .logo3 {{
          background-image: url("data:image/jpeg;base64,{img_str6}");
          background-size: contain;
          background-repeat: no-repeat;
          width: 100px;
          height: 45px;
          margin-right: 20px;
        }}
      .logo2 {{
          background-image: url("data:image/jpeg;base64,{img_str3}");
          background-size: contain;
          background-repeat: no-repeat;
          width: 60px;
          height: 45px;
      }}
      .logo1 {{
          background-image: url("data:image/jpeg;base64,{img_str2}");
          background-size: contain;
          background-repeat: no-repeat;
          width: 80px;
          height: 35px;
          margin-right: 20px;
      }}
      .logo {{
          background-image: url("data:image/jpeg;base64,{img_str1}");
          background-size: contain;
          background-repeat: no-repeat;
          width: 70px;
          height: 35px;
      }}
      .main-content {{
          display: flex;
          justify-content: space-between;
          border-radius: 10px;
          margin-top: 60px;
      }}
      .header .title h1 {{
      }}
      .fotodas{{
          display: flex;
          justify-content: space-between;
      }}
      .des{{
          flex: 1;
          margin-left: 10px;
          text-align: justify;
      }}
      .description {{
          border-radius: 10px;
          margin-top: 20 px;
      }}
      .air {{
          flex: 1;
          background-image: url("data:image/jpeg;base64,{img_str4}");
          background-size: cover;
          background-repeat: no-repeat;
          margin-top: 20 px;
          border-radius: 10px;
      }}
      .objective {{
          flex: 1;
          padding: 20px;
          border-radius: 10px;
          background-color: #D9FAEE;
          margin-right: 10px;
          width: 50%;
      }}
      .benefit {{
          flex: 1;
          padding: 20px;
          border-radius: 10px;
          background-color: #D9FAEE;
          width: 50%;
      }}
      </style>
      """,
      unsafe_allow_html=True
  )

  st.markdown(
  """
      <div class="header">
          <div class="title">
              <h1>Water Quality</h1>
          </div>
          <div class="kumlogo">
              <div class="logo2"></div>
              <div class="logo3"></div>
              <div class="logo1"></div>
              <div class="logo"></div>
          </div>
      </div>
      <hr style="margin: 20px 0; border-top: 1px solid #ddd;">
      <div class="main-content">
          <div class="description">
            <div class = "fotodas">
              <div class ="air"></div>
              <p class = "des">Pentingnya kualitas air dalam menjaga keberlanjutan lingkungan dan kesehatan masyarakat telah menjadi perhatian utama di seluruh dunia. 
              Air adalah aset berharga yang tidak dapat digantikan, namun sering kali terpapar oleh berbagai faktor, mulai dari polusi industri hingga limbah domestik. 
              Dalam menghadapi tantangan ini, perlu adanya upaya untuk mengembangkan sistem prediktif yang mampu memantau dan mengukur kualitas air secara akurat. 
              Melalui analisis data dan teknologi yang inovatif, kita dapat mengidentifikasi pola perilaku air dan memprediksi potensi risiko yang terkait dengan perubahan lingkungan.</p>
            </div>
              <p style="text-align: justify;">Pada Studi Kasus Water Quality Prediction bertujuan untuk mengatasi tantangan ini dengan mengembangkan model prediksi yang efektif dan dapat diandalkan. 
              Dengan memanfaatkan data terkini dan teknik analisis yang canggih, kita dapat memperkirakan kualitas air di lokasi tertentu dan mengidentifikasi faktor-faktor yang berpotensi mempengaruhi. 
              Dengan demikian, upaya ini tidak hanya akan membantu dalam menjaga keberlanjutan sumber daya air, tetapi juga dapat memberikan informasi yang berharga bagi pengambil keputusan dalam menangani masalah lingkungan di masa depan.</p>
              <div style='display: flex; justify-content: space-between; margin-bottom: 20px;'>
                <div class="objective">
                  <h6 style="text-align: center;">Objective</h6>
                  <p style="font-size: 0.8em; text-align: justify;">
                      <span><b>1. Meningkatkan pemahaman tentang kualitas air</b></span><br>
                      Melalui analisis prediktif, tujuan utama adalah meningkatkan pemahaman tentang faktor-faktor yang memengaruhi kualitas air di berbagai lokasi. Hal ini akan membantu dalam mengidentifikasi sumber polusi dan potensi risiko terhadap kesehatan manusia dan lingkungan.<br>
                      <span><b>2. Peningkatan responsibilitas lingkungan</b></span><br>
                      Dengan memprediksi kualitas air secara akurat, tujuan ini adalah untuk memberikan solusi yang dapat meningkatkan tanggung jawab lingkungan dalam pengelolaan sumber daya air. Hal ini dapat mencakup upaya untuk mengurangi polusi air, mengoptimalkan penggunaan air, dan meminimalkan dampak negatif terhadap ekosistem air.<br>
                      <span><b>3. Mendukung keberlanjutan lingkungan</b></span><br>
                      Melalui pemahaman yang lebih baik tentang kualitas air, tujuan ini adalah untuk mendukung upaya-upaya dalam menjaga keberlanjutan lingkungan. Hal ini dapat mencakup pengelolaan air yang lebih efisien, perlindungan habitat air, dan pemulihan ekosistem air yang terganggu.<br>
                      <span><b>4. Peningkatan kesehatan masyarakat</b></span><br>
                      Dengan memantau kualitas air secara berkala dan melakukan prediksi yang akurat, tujuan ini adalah untuk melindungi kesehatan masyarakat dari risiko yang berkaitan dengan konsumsi air yang tercemar. Hal ini akan membantu dalam mengurangi risiko penyakit terkait air dan meningkatkan kualitas hidup masyarakat secara keseluruhan.
                  </p>
                </div>
              </div>
          </div>
      </div>
  """, 
  unsafe_allow_html=True
  )

  # Footer
  st.markdown("""
  <link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
  <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-top: 40px; border-top: 1px solid #ddd; padding-top: 10px;">
      <div style="display: flex; align-items: flex-start; font-size: 15px;">
          <i class="material-icons" style="font-size: 25px; margin-right: 5px; color: #4C4D50 ;">location_on</i>
          <div style="font-size: 15px; color: #4C4D50;">
              Jl. Duren Tiga Raya No.09, RT.12/RW.1, Duren<br>
              Tiga, Kec. Pancoran, Kota Jakarta Selatan,<br>
              Daerah Khusus Ibukota Jakarta 12760
          </div>
      </div>
  </div>
  """, unsafe_allow_html=True)

  st.markdown("""
  <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css" rel="stylesheet">
  <div style="display: flex; flex-direction: column;">
      <hr style="border-top: 1px solid #ddd; margin: 10px 0;">
      <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 5px;">
          <div style="display: flex; align-items: center; font-size: 15px; color: #4C4D50;">
              <i class="far fa-copyright" style="font-size: 20px; margin-right: 5px;"></i>
              2024 <span style="margin-left: 2px"><b>Fun-tastic Four</b></span>. All Rights Reserved
          </div>
          <div style="font-size: 14px; margin: 0; color: #4C4D50;">
              SIB Cycle 6 | 2024
          </div>
      </div>
  </div>
  """, unsafe_allow_html=True)

# Data Description Page
if selected == 'Data Description':

  # Load the image background
  image_path = os.path.join(working_dir, "Foto/background.png")
  image = Image.open(image_path)
  buffered = io.BytesIO()
  image.save(buffered, format="PNG")
  img_str5 = base64.b64encode(buffered.getvalue()).decode()

  # Set the background image
  st.markdown(
      f"""
      <style>
      .stApp {{
          background-image: url("data:image/jpeg;base64,{img_str5}");
          background-size: cover;
          background-position: center;
      }}
      </style>
      """,
      unsafe_allow_html=True
  )

  st.markdown('<h1>Dataset Description</h1>', unsafe_allow_html=True)
  st.markdown('---')
  st.markdown("""<div style="text-align:justify;"> Dataset ini berasal dari link <a href="https://www.kaggle.com/competitions/tfug-mysuru-water-quality-prediction/data">https://www.kaggle.com/competitions/tfug-mysuru-water-quality-prediction/data</a>. 
  File yang digunakan adalah train.csv yang berisi beberapa kolom diantaranya id unik setiap baris, 6 kolom category dengan akhiran A sampai F, 
  9 kolom feature dengan akhiran A sampai I, 10 kolom composition dengan akhiran A sampai J, kolom unit yang merupakan satuan ukuran hasil, 
  dan kolom result yang merupakan ukuran kualitas air sebagai nilai yang akan diprediksi. Berikut adalah penjelasan masing-masing kolom.

  1. Kolom category adalah berbagai fitur kategoris untuk suatu titik data seperti negara pengumpulan data, situs pengumpulan data, media sampel, dan lainnya. Kolom category ini terdiri dari categoryA, categoryB, categoryC, categoryD, categoryE, dan categoryF.
  2. Kolom feature adalah berbagai fitur demografi yang mempengaruhi pencemaran air di suatu wilayah tertentu seperti kepadatan penduduk, PDB, kekeringan di suatu wilayah, angka melek huruf siswa di suatu wilayah, dan lainnya. Kolom feature ini terdiri dari featureA, featureB, featureC, featureD, featureE, featureF, featureG, featureH, dan featureI.
  3. Kolom composition adalah susunan berbagai unsur seperti kertas, sampah plastik, karton, dan lain-lain di dalam air. Kolom composition ini terdiri dari compositionA, compositionB, compositionC, compositionD, compositionE, compositionF, compositionG, compositionH, compositionI, dan compositionJ.
  4. Kolom unit adalah satuan ukuran yang digunakan untuk mengukur nilai hasil.
  5. Kolom result adalah nilai yang menyatakan kualitas air berdasarkan berbagai faktor yang tersedia dalam kumpulan data.
  6. Kolom water quality adalah kategori dari kualitas air berdasarkan nilai result yang dikelompokkan menjadi 5 kategori, nilai result dibawah = 0.2 kualitas air kategori very poor (sangat buruk), nilai result dibawah = 0.4 kualitas air kategori poor (buruk), 
  nilai result dibawah = 0.6 kualitas air kategori standard (sedang/cukup), nilai result dibawah = 0.8 kualitas air kategori good (bagus), dan nilai result diatas = 0.8 kualitas air kategori very good (sangat bagus).
  </div>""", unsafe_allow_html=True)
  st.markdown('<div>Berikut Dataset yang digunakan dalam prediksi kualitas air ini :</div>', unsafe_allow_html=True)

  df = load_data()

  st.markdown("""
    <style>
    .stSelectbox div[data-baseweb="select"] > div {
        background-color: #ffffff;
    }
    </style>
  """, unsafe_allow_html=True)

  # Dropdownlist untuk memilih kategori
  option = st.selectbox('Select Category:', ['All Data', 'Very Poor', 'Poor', 'Standard', 'Good','Very Good'])

  # Filter data berdasarkan kategori yang dipilih
  if option == 'All Data':
      st.dataframe(df.sort_values(by='water_quality'), width=1050)
      total_rows = len(df)
      total_columns = len(df.columns)
      st.write(f"Total data terdiri dari {total_rows:,} baris dan {total_columns} kolom")
  elif option == 'Very Poor':
      verypoor_df = df[df['water_quality'] == 'Very Poor'].sort_values(by='water_quality')
      st.dataframe(verypoor_df, width=1050)
      total_rows = len(verypoor_df)
      total_columns = len(verypoor_df.columns)
      st.write(f"Total data Very Poor terdiri dari {total_rows:,} baris dan {total_columns} kolom")
  elif option == 'Poor':
      poor_df = df[df['water_quality'] == 'Poor'].sort_values(by='water_quality')
      st.dataframe(poor_df, width=1050)
      total_rows = len(poor_df)
      total_columns = len(poor_df.columns)
      st.write(f"Total data Poor terdiri dari {total_rows:,} baris dan {total_columns} kolom")
  elif option == 'Standard':
      standard_df = df[df['water_quality'] == 'Standard'].sort_values(by='water_quality')
      st.dataframe(standard_df, width=1050)
      total_rows = len(standard_df)
      total_columns = len(standard_df.columns)
      st.write(f"Total data Standard terdiri dari {total_rows:,} baris dan {total_columns} kolom")
  elif option == 'Good':
      good_df = df[df['water_quality'] == 'Good'].sort_values(by='water_quality')
      st.dataframe(good_df, width=1050)
      total_rows = len(good_df)
      total_columns = len(good_df.columns)
      st.write(f"Total data Good terdiri dari {total_rows:,} baris dan {total_columns} kolom")
  else:
      verygood_df = df[df['water_quality'] == 'Very Good'].sort_values(by='water_quality')
      st.dataframe(verygood_df, width=1050)
      total_rows = len(verygood_df)
      total_columns = len(verygood_df.columns)
      st.write(f"Total data Very Good terdiri dari {total_rows:,} baris dan {total_columns} kolom")

# Analytics Page 
if selected == 'Analytics':

    # Load the image background
    image_path = os.path.join(working_dir, "Foto/background.png")
    image = Image.open(image_path)
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str5 = base64.b64encode(buffered.getvalue()).decode()

    # Set the background image
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpeg;base64,{img_str5}");
            background-size: cover;
            background-position: center;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    df = load_data()
    df2 = load_data2()

    # page title
    st.title('Data Analysis Visualization')
    with st.expander('Dataset Describe', expanded=False):
      st.dataframe(df2.describe(), width=1050)

    col1, col2 = st.columns(2)
    with col1:
      st.markdown("""
          <style>
          div[data-baseweb="select"] > div {
              background-color: #ffffff;
          }
          </style>
      """, unsafe_allow_html=True)

      selected_view = st.selectbox("Select Distribution:", ["CategoryA Distribution", "Unit Distribution"])
      if selected_view == "CategoryA Distribution":
        category_counts = df['categoryA'].value_counts()
        threshold_percentage = 1.00 / 100
        threshold = threshold_percentage * len(df)
        df['category_A'] = df['categoryA'].apply(
            lambda x: x if category_counts[x] >= threshold else 'Other')
        fig1 = px.pie(df, names='category_A', title='CategoryA Distribution')
        st.plotly_chart(fig1, use_container_width=True)
      
      elif selected_view == "Unit Distribution":
        #st.subheader('Distribusi Unit')
        fig2 = px.pie(df, names='unit', title='Unit Distribution')
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
      st.markdown("""
          <style>
          div[data-baseweb="select"] > div {
              background-color: #ffffff;
          }
          </style>
      """, unsafe_allow_html=True)

      selected_view = st.selectbox("Select Distribution:", ["Water Quality Distribution"])      
      if selected_view == "Water Quality Distribution":
        fig3 = px.pie(df, names='water_quality',color='water_quality', title='Water Quality Distribution',
                      color_discrete_map={
                      'Very Poor': '#A70A05',
                      'Poor': '#FE0901',
                      'Standard': '#D4AC0D',
                      'Good': '#58D68D',
                      'Very Good': '#052EAF'
            }
        )
        st.plotly_chart(fig3, use_container_width=True)

    fig5 = px.bar(df, x='categoryC', y=['featureD', 'featureE', 'featureF', 'compositionC'], title='Bar Chart: CategoryC by FeatureD,FeatureE,FeatureF,CompositionC')
    st.plotly_chart(fig5, use_container_width=True)

    fig6 = px.bar(df, x='category_A', y=['featureD', 'featureE', 'featureF', 'compositionC'], title='Bar Chart: CategoryA by FeatureD, FeatureE, FeatureF, CompositionC')
    st.plotly_chart(fig6, use_container_width=True)

# Water Prediction Page
if selected == 'Water Prediction':

    # Load the image background
    image_path = os.path.join(working_dir, "Foto/background.png")
    image = Image.open(image_path)
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str5 = base64.b64encode(buffered.getvalue()).decode()

    # Set the background image
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpeg;base64,{img_str5}");
            background-size: cover;
            background-position: center;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    # Function to prepare the data for modeling
    def model_prepare(df_modeller, selected_features, target_feature):
        X = df_modeller[selected_features]
        y = df_modeller[target_feature]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        return X_train, X_test, y_train, y_test

    # Load data
    selected_features = ['categoryA', 'categoryC', 'featureD', 'featureE', 'featureF', 'compositionC', 'unit']
    target_feature = 'result'

    # Prepare data
    df_modeller = load_data3()
    X_train, X_test, y_train, y_test = model_prepare(df_modeller, selected_features, target_feature)

    # Function to calculate model evaluation metrics
    def evaluate_model(model, X, y):
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        return r2, mae, mse, rmse

    # Calculate model evaluation metrics
    r2_score, mae, mse, rmse = evaluate_model(water_model, X_test, y_test)

    # Streamlit app
    st.markdown(
        """
        <style>
        .title1 {
            font-size: 32px !important;
        }
        </style>
        """, 
        unsafe_allow_html=True
    )
    st.markdown("<h1 class='title1'>Accuracy in The Best Model (Random Forest Tuning Hyperparameters)</h1>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div style='display: flex; justify-content: space-between; margin-bottom: 20px;'>
            <div style='background-color: #78BDB0; padding: 10px; border-radius: 5px; margin-right: 10px; flex-grow: 1; text-align: center;'>
                <strong style='color: white; display: block;'>R2 Score</strong>
                <span style='color: white; font-size: 40px;'>{r2_score:.3f}</span>
            </div>
            <div style='background-color: #78BDB0; padding: 10px; border-radius: 5px; margin-right: 10px; flex-grow: 1; text-align: center;'>
                <strong style='color: white; display: block;'>MAE</strong>
                <span style='color: white; font-size: 40px;'>{mae:.3f}</span>
            </div>
            <div style='background-color: #78BDB0; padding: 10px; border-radius: 5px; margin-right: 10px; flex-grow: 1; text-align: center;'>
                <strong style='color: white; display: block;'>MSE</strong>
                <span style='color: white; font-size: 40px;'>{mse:.3f}</span>
            </div>
            <div style='background-color: #78BDB0; padding: 10px; border-radius: 5px; flex-grow: 1; text-align: center;'>
                <strong style='color: white; display: block;'>RMSE</strong>
                <span style='color: white; font-size: 40px;'>{rmse:.3f}</span>
            </div>
        </div>
        """, 
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <style>
        .title {
            font-size: 26px !important;
        }
        </style>
        """, 
        unsafe_allow_html=True
    )
    st.markdown("<h1 class='title'>Water Quality Prediction</h1>", unsafe_allow_html=True)

    # getting the input data from the user
    col1, space1, col2, space2, col3 = st.columns([1, 0.1, 1, 0.1, 1])

    with col1:
        categoryA = st.slider('CategoryA', min_value=0, max_value=166, format="%f")
        featureE = st.slider('FeatureE', min_value=6.4586, max_value=38.7672, format="%f")
        unit = st.slider('Unit', min_value=0, max_value=18, format="%f")

    with col2:
        categoryC = st.slider('CategoryC', min_value=0, max_value=2211, format="%f")
        featureF = st.slider('FeatureF', min_value=0, max_value=100, format="%f")

    with col3:
        featureD = st.slider('FeatureD', min_value=18.1485, max_value=377.3796, format="%f")
        compositionC = st.slider('CompositionC', min_value=0.0, max_value=44.05, format="%f")

    # Code for Prediction
    water_quality_diagnosis = ''

    # creating a button for Prediction
    if st.button('Water Quality Prediction Result'):
        user_input = np.array([
            categoryA,	categoryC,	featureD,	featureE,	featureF,	compositionC,	unit
        ])
        user_input_array = [user_input]
        water_prediction = water_model.predict(user_input_array)
        water_quality_diagnosis_array = []

        for prediction in water_prediction:
          if prediction <= 0.2:
              diagnosis = f'Based on the new parameters, the water quality is Very Poor ({prediction})'
              background_color = 'background-color: #A70A05;'
              color = 'color: white;'
          elif prediction <= 0.4:
              diagnosis = f'Based on the new parameters, the water quality is Poor ({prediction})'
              background_color = 'background-color: #FE0901;'
              color = 'color: white;'
          elif prediction <= 0.6:
              diagnosis = f'Based on the new parameters, the water quality is Standard ({prediction})'
              background_color = 'background-color: #D4AC0D;'
              color = 'color: white;'
          elif prediction <= 0.8:
              diagnosis = f'Based on the new parameters, the water quality is Good ({prediction})'
              background_color = 'background-color: #58D68D;'
              color = 'color: white;'
          else:
              diagnosis = f'Based on the new parameters, the water quality is Very Good ({prediction})'
              background_color = 'background-color: #052EAF;'
              color = 'color: white;'
        
          water_quality_diagnosis_array.append((diagnosis, background_color, color))

        for diagnosis, background_color, color in water_quality_diagnosis_array:
          st.markdown(f"<div style='padding: 10px; {background_color}; {color}'>{diagnosis}</div>", unsafe_allow_html=True)

# About Us Page
if selected == 'About Us':

    # Load the image background
    image_path = os.path.join(working_dir, "Foto/background.png")
    image = Image.open(image_path)
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str5 = base64.b64encode(buffered.getvalue()).decode()

    # Set the background image
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpeg;base64,{img_str5}");
            background-size: cover;
            background-position: center;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
    
    st.title('About Us')
    st.markdown('---')
 #   from PIL import Image

    team_members = [
        {
            "name": "Salsabila Nur Yasmin",
            "role": "Mentor",
            "image_id": os.path.join(working_dir, "Foto/salsa.png"),
            "description": "Salsa is an experienced data scientist with over 2 years of experience in the industry. She has mentored numerous teams and helped them achieve their goals."
        },
        {
            "name": "Lutfi Julpian",
            "role": "Project Leader",
            "image_id": os.path.join(working_dir, "Foto/lutfi.png"),
            "description": "Lutfi leads the project with a strong vision and excellent management skills. He ensures the project stays on track and meets all deadlines."
        },
        {
            "name": "Vindiani Nora Putri",
            "role": "Data Analyst",
            "image_id": os.path.join(working_dir, "Foto/Vindi.png"),
            "description": "Vindi is a data analyst with a knack for uncovering insights from complex datasets. She specializes in statistical analysis and data modeling."
        },
        {
            "name": "Jauza Hayah Anbari",
            "role": "Data Visualization",
            "image_id": os.path.join(working_dir, "Foto/Jauza.png"),
            "description": "Jauza is responsible for creating intuitive and informative data visualizations. She helps transform raw data into compelling stories."
        },
        {
            "name": "I Kadek Ananda Krisna Wiralaksana",
            "role": "Modeller",
            "image_id": os.path.join(working_dir, "Foto/nanda.png"),
            "description": "Nanda builds predictive models that help us understand and forecast trends. He is skilled in machine learning and AI technologies."
        }
    ]

    image_width = 150
    image_height = 200

    for member in team_members:
        cols = st.columns([1, 4])
        if 'image_id' in member:
            image = Image.open(member["image_id"])
            image = image.resize((image_width, image_height))
            cols[0].image(image, use_column_width=False)
            st.write('<style>div.Widget.row-widget.stRadio>div{flex-direction:column;}</style>', unsafe_allow_html=True)
        
        with cols[1]:
            st.markdown(f"""
            <div style='margin-top: 25px; background-color: #D9FAEE; padding: 10px; border-radius: 5px;'>
                <strong>{member['name']}</strong>
                <br>
                <em>{member['role']}</em>
                <br><br>
                {member['description']}
            </div>
            """, unsafe_allow_html=True)
