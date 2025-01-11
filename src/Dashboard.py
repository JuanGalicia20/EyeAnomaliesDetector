import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import cv2
import os
import plotly.graph_objects as go
import plotly.express as px
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from io import BytesIO
from datetime import datetime
import base64
from PIL import Image
from reportlab.lib.utils import ImageReader

# Configuración de la página
st.set_page_config(
    page_title="Diagóstico Preliminar RD",
    layout="wide",
    initial_sidebar_state="collapsed"
)

def load_sample_image(image_path):
    if os.path.exists(image_path):
        return Image.open(image_path)
    else:
        st.error(f"No se pudo encontrar la imagen de muestra: {image_path}")
        return None

# Función de preprocesamiento actualizada
def preprocess_macular_retinopathy(img, scale=600):
    def scaleRadius(img, scale):
        x = img[img.shape[0] // 2, :, :].sum(1)
        r = (x > x.mean() / 10).sum() / 2
        s = scale * 1.0 / r
        return cv2.resize(img, (0, 0), fx=s, fy=s)

    def preprocess_image(img):
        # Seleccionar el canal verde de la imagen
        img_green = img[:, :, 1]

        # Aplicar CLAHE al canal verde
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_clahe = clahe.apply(img_green.astype(np.uint8))

        # Restar el color promedio local
        img_clahe = cv2.addWeighted(img_clahe, 4, cv2.GaussianBlur(img_clahe, (0, 0), scale / 30), -4, 128)

        return img_clahe

    if img is None:
        return None

    try:
        # Escalar la imagen
        img = scaleRadius(img, scale)

        # Aplicar preprocesamiento
        img = preprocess_image(img)

        # Eliminar el 10% exterior
        mask = np.zeros(img.shape, dtype=np.uint8)
        cv2.circle(mask, (img.shape[1] // 2, img.shape[0] // 2), int(scale * 0.9), (1, 1, 1), -1, 8, 0)
        img = img * mask + 128 * (1 - mask)

        # Asegurarse de que la salida tenga 3 canales y tamaño 224x224
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img = cv2.resize(img, (600, 600))

        return img

    except Exception as e:
        st.error(f"Error en el preprocesamiento: {str(e)}")
        return None

def predict_image(model, img_array):
    try:
        # Convertir de RGB a BGR si es necesario
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Preprocesar
        preprocessed_img = preprocess_macular_retinopathy(img_array)
        if preprocessed_img is None:
            return None, None, None, None, None

        # Ajustar el tamaño de la imagen a 600x600 sin normalizar
        preprocessed_img = cv2.resize(preprocessed_img, (600, 600))

        # Expandir dimensiones para que el modelo reciba (1, 600, 600, 3)
        img_array_expanded = np.expand_dims(preprocessed_img, axis=0)

        # Predicción
        predictions = model.predict(img_array_expanded)
        predicted_class = np.argmax(predictions, axis=1)

        # Calcular la probabilidad binaria comparando clase 0 vs suma de las demás
        prob_class_0 = predictions[0, 0]  # Probabilidad de la clase 0 (No RD)
        prob_other_classes = np.sum(predictions[0, 1:])  # Suma de probabilidades de las demás clases
        
        # Determinar predicción binaria (0 si clase 0 es mayor, 1 si la suma de las otras es mayor)
        binary_predictions = 1 if prob_other_classes > prob_class_0 else 0
        
        # La probabilidad binaria será la suma de las probabilidades de las clases positivas
        y_pred_prob_binary = prob_other_classes

        return binary_predictions, y_pred_prob_binary, predicted_class, predictions, preprocessed_img

    except Exception as e:
        st.error(f"Error en la predicción: {str(e)}")
        return None, None, None, None, None

def create_binary_probability_chart(prob):
    # Convertir probabilidad a porcentaje
    prob_percentage = prob * 100
    no_rd_percentage = (1 - prob) * 100
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=['No Retinopatía', 'Retinopatía'],
        y=[no_rd_percentage, prob_percentage],
        marker_color=['#2ecc71', '#e74c3c'],
        text=[f'{no_rd_percentage:.1f}%', f'{prob_percentage:.1f}%'],
        textposition='auto',
    ))
    fig.update_layout(
        title={
            'text': 'Probabilidad de Retinopatía Diabética',
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        yaxis_title='Probabilidad (%)',
        yaxis=dict(range=[0, 100]),  # Establecer rango de 0 a 100
        xaxis_title='Clase',
        showlegend=False,
        height=400,
        template='plotly_white'
    )
    return fig

def create_multiclass_probability_chart(probs):
    labels = ['No RD', 'Leve', 'Moderada', 'Severa', 'Proliferativa']
    # Convertir probabilidades a porcentajes
    percentages = [prob * 100 for prob in probs[0]]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels,
        y=percentages,
        marker_color=['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c', '#c0392b'],
        text=[f'{p:.1f}%' for p in percentages],
        textposition='auto',
    ))
    fig.update_layout(
        title={
            'text': 'Probabilidad por Nivel de Severidad',
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        yaxis_title='Probabilidad (%)',
        yaxis=dict(range=[0, 100]),  # Establecer rango de 0 a 100
        xaxis_title='Nivel de Severidad',
        showlegend=False,
        height=400,
        template='plotly_white'
    )
    return fig

def create_pdf(image_pil, preprocessed_img, binary_pred, multi_pred, predicted_class):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    
    # Título
    c.setFont("Helvetica-Bold", 24)
    c.drawString(50, height - 50, "Reporte de Diagnóstico - Retinopatía Diabética")
    
    # Información del paciente
    c.setFont("Helvetica", 12)
    c.drawString(50, height - 100, f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Guardar imágenes temporalmente para el PDF
    img_buffer = BytesIO()
    image_pil.save(img_buffer, format='PNG')
    img_buffer.seek(0)
    
    # Convertir la imagen preprocesada a formato PIL y guardarla
    preprocessed_buffer = BytesIO()
    if isinstance(preprocessed_img, np.ndarray):
        preprocessed_pil = Image.fromarray(preprocessed_img)
        preprocessed_pil.save(preprocessed_buffer, format='PNG')
        preprocessed_buffer.seek(0)
    
    # Redimensionar y colocar imágenes
    c.drawImage(ImageReader(img_buffer), 50, height - 350, width=200, height=200)
    if isinstance(preprocessed_img, np.ndarray):
        c.drawImage(ImageReader(preprocessed_buffer), 300, height - 350, width=200, height=200)
    
    # Resultados
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, height - 400, "Resultados del Análisis:")
    
    c.setFont("Helvetica", 12)
    result_text = "No se detectó Retinopatía" if predicted_class == 0 else "Se detectaron signos de Retinopatía"
    c.drawString(50, height - 430, f"Diagnóstico: {result_text}")
    
    # Probabilidades
    c.drawString(50, height - 460, f"Probabilidad de Retinopatía: {binary_pred:.2%}")
    
    # Severidad
    severity_levels = ['No RD', 'Leve', 'Moderada', 'Severa', 'Proliferativa']
    y_position = height - 490
    c.drawString(50, y_position, "Probabilidades por nivel de severidad:")
    for i, (level, prob) in enumerate(zip(severity_levels, multi_pred[0])):
        y_position -= 20
        c.drawString(70, y_position, f"{level}: {prob:.2%}")
    
    # Logos
    #logo_positions = [
    #    ("./static/img/uvg.png", 50, 100, 100),
    #    ("./static/img/ceia.png", width/2 - 100, 100, 150),
    #    ("./static/img/uno.png", width - 150, 100, 100)
    #]
    
    #for logo_path, x, y, w in logo_positions:
    #    if os.path.exists(logo_path):
    #        try:
    #            c.drawImage(logo_path, x, y, width=w, preserveAspectRatio=True)
    #        except Exception as e:
    #            print(f"Error al cargar el logo {logo_path}: {str(e)}")
    
    # Disclaimer
    c.setFont("Helvetica", 8)  # Cambiado de Helvetica-Italic a Helvetica
    c.drawString(50, 50, "Este reporte es generado automáticamente y debe ser validado por un profesional médico.")
    
    c.save()
    buffer.seek(0)
    return buffer


# Cargar el modelo
@st.cache_resource
def load_keras_model():
    return load_model("./static/model/vgg19_2_11_v2.keras")



# Estilo CSS mejorado
st.markdown("""
    <style>
        /* Estilos generales */
        .main {
            padding: 2rem;
        }
        
        /* Header container */
        .header-container {
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            padding: 2rem;
            border-radius: 1rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 2rem;
        }
        
        /* Botones */
        .stButton > button {
            width: 100% !important;
            height: 3.5rem !important;
            background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
            border: 2px solid #e9ecef;
            border-radius: 0.5rem;
            transition: all 0.3s ease;
            font-weight: 500;
            margin: 0.5rem 0;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
            border-color: #4CAF50;
        }
        
        /* Versión tag */
        .version-tag {
            color: #6c757d;
            font-size: 0.9rem;
            font-weight: 500;
            text-align: center;
            margin-top: 0.5rem;
        }
        
        /* Contenedor de desarrollador */
        .developer-container {
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 0.5rem;
            text-align: center;
            margin: 1rem 0;
            border: 1px solid #e9ecef;
        }
        
        /* Divisores */
        .custom-divider {
            margin: 2rem 0;
            border: none;
            height: 1px;
            background: linear-gradient(90deg, transparent, #dee2e6, transparent);
        }
        
        /* Cards para resultados */
        .result-card {
            background: white;
            padding: 1.5rem;
            border-radius: 0.5rem;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
            border: 1px solid #e9ecef;
        }
        
        /* Footer */
        .footer-container {
            background-color: #f8f9fa;
            padding: 1.5rem;
            border-radius: 0.5rem;
            margin-top: 2rem;
            text-align: center;
        }
        
        /* Contenedor de logos */
        .logo-container {
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 2rem;
            margin-top: 1rem;
        }
        
        /* File uploader */
        .uploadedFile {
            border: 2px dashed #4CAF50;
            border-radius: 0.5rem;
            padding: 1rem;
            margin: 1rem 0;
        }
        
        /* Alerts */
        .stAlert {
            border-radius: 0.5rem;
            border: none;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        }
    </style>
""", unsafe_allow_html=True)

# Header con diseño mejorado
st.markdown("""
    <div class="header-container">
        <h1 style='text-align: center; color: #2c3e50; margin-bottom: 0.5rem;'>
            🏥 OptiMIRA: Monitoreo Inteligente para la Detección Temprana de Retinopatía Diabética
        </h1>
        <p class="version-tag">Versión del modelo: 1.0.0</p>
    </div>
""", unsafe_allow_html=True)

with st.container():
    col1, col2, col3 = st.columns(3)
    
    with col1:
        info_button = st.button("ℹ️ Información del Modelo", key="info_button")

    with col2:
        # Luego el botón
        license_button = st.button("📄 Licencia", key="license_button")
        st.markdown("""
            <div style="text-align: center; margin-bottom: 1rem;">
                <p style="margin: 0;">Desarrollado por:</p>
                <strong>Juan Andrés Galicia Reyes</strong>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        act_button = st.button("🔍 Actualizaciones", key="updates_button")

if info_button:
    st.info("""
    **Modelo de Deep Learning para Detección de Retinopatía**
    
    - Entrenado con miles de imágenes
    - Validado por oftalmólogos expertos
    - Actualización continua del modelo
    - Precisión superior al 90%
    """)

if act_button:
    st.info("""
    **Próximas Mejoras**
            
    - Mayor precisión en detección
    - Expansión a nuevas enfermedades
    - Mejoras en la interfaz
    - Mejoras en el sistema de generación de PDF a correos automáticos
    """)

if license_button:
    st.info("""
    **Licencia de Software (SW License)**

    Copyright (c) 2025 Juan Andrés Galicia Reyes

    Por la presente se otorga permiso, de forma gratuita, a cualquier persona que obtenga una copia de este software y los archivos de documentación asociados (el "Software"), para utilizar el Software sin restricciones, incluyendo, sin limitación, los derechos para:

    • Usar el software en entornos clínicos y de investigación
    • Estudiar cómo funciona el software y adaptarlo a sus necesidades específicas
    • Redistribuir el software con fines no comerciales
    • Mejorar el software y compartir las mejoras con la comunidad

    **Condiciones:**

    1. **Atribución:** Debe proporcionar atribución adecuada al autor original, incluyendo un enlace a la licencia y indicando si se realizaron cambios.

    2. **No Comercial:** No puede utilizar este software con fines comerciales sin el permiso expreso del autor.

    3. **Compartir Igual:** Si remezcla, transforma o crea a partir del material, debe distribuir sus contribuciones bajo la misma licencia que el original.

    4. **Sin Garantía:** El software se proporciona "tal cual", sin garantía de ningún tipo, expresa o implícita. El autor no será responsable de ningún daño o reclamación en relación con el software.

    **Limitación de Responsabilidad:**

    • Este software está diseñado como una herramienta de apoyo y no debe utilizarse como único medio de diagnóstico.
    • Las predicciones y análisis generados por el software deben ser validados por profesionales médicos cualificados.
    • El autor no se hace responsable de diagnósticos erróneos o decisiones médicas basadas únicamente en los resultados del software.

    **Uso en Investigación:**

    Si utiliza este software en investigación académica, por favor cite:

    Galicia-Reyes, J.A. (2025). Sistema de Detección de Retinopatía Diabética mediante Deep Learning.
    Universidad del Valle de Guatemala.

    Para cualquier consulta sobre licencias comerciales o colaboraciones, contactar a: juanandresgaliciareyes@gmail.com
    """)
# Divisor personalizado
st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)


# Uploader de imagen
st.markdown("""
    <div style='background-color: #f8f9fa; padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem;'>
        <h3 style='color: #2c3e50; margin-bottom: 1rem;'>📸 Cargar Imagen</h3>
    </div>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["📤 Subir Imagen", "🔍 Usar Imagen de Muestra"])

with tab1:
    uploaded_file = st.file_uploader("Seleccione una imagen de fondo de ojo", type=["jpg", "jpeg", "png"])
    
with tab2:
    st.markdown("### Seleccione una imagen de prueba")
    # Contenedor para los botones de imágenes de prueba
    sample_images_col1, sample_images_col2, sample_images_col3, sample_images_col4, sample_images_col5 = st.columns(5)
    
    # Rutas a las imágenes de prueba (ajusta según tu estructura de archivos)
    sample_images = {
        "Muestra 1 (Sano)": "./static/img/samples/sample-sano.png",
        "Muestra 2 (Retinopatía Leve)": "./static/img/samples/sample-dr-leve.jpeg",
        "Muestra 3 (Retinopatía Moderada)": "./static/img/samples/sample-dr-moderada.jpg",
        "Muestra 4 (Retinopatía Grave)": "./static/img/samples/sample-dr-grave.png",
        "Muestra 5 (Retinopatía Proliferativa)": "./static/img/samples/sample-dr-prolif.png"
    }
    
    # Variable para almacenar la imagen seleccionada
    selected_sample = None
    
    # Crear los botones en las columnas
    with sample_images_col1:
        if st.button("Muestra 1 (Sano)", use_container_width=True):
            selected_sample = "Muestra 1 (Sano)"
            
    with sample_images_col2:
        if st.button("Muestra 2 (Retinopatía Leve)", use_container_width=True):
            selected_sample = "Muestra 2 (Retinopatía Leve)"
            
    with sample_images_col3:
        if st.button("Muestra 3 (Retinopatía Moderada)", use_container_width=True):
            selected_sample = "Muestra 3 (Retinopatía Moderada)"
    
    with sample_images_col4:
        if st.button("Muestra 4 (Retinopatía Grave)", use_container_width=True):
            selected_sample = "Muestra 4 (Retinopatía Grave)"
            
    with sample_images_col5:
        if st.button("Muestra 5 (Retinopatía Proliferativa)", use_container_width=True):
            selected_sample = "Muestra 5 (Retinopatía Proliferativa)"
    
    # Mostrar previsualización de la imagen seleccionada
    if selected_sample:
        image_path = sample_images[selected_sample]
        sample_image = load_sample_image(image_path)

# Procesar la imagen (ya sea cargada o de muestra)
if uploaded_file is not None or (selected_sample and sample_image):
    try:
        modelo = load_keras_model()
        
        # Procesar imagen
        if uploaded_file is not None:
            image_pil = Image.open(uploaded_file)
        else:
            image_pil = sample_image
        image_array = np.array(image_pil)
        
        if modelo:
            with st.spinner('Procesando imagen...'):
                predicted_class, predicted_proba, predicted_multiclass, predicted_proba_multiclass, img_pre = predict_image(modelo, image_array)
            
            if predicted_class is not None and img_pre is not None:
                # Primera fila: Imágenes (más pequeñas)
                st.subheader("Análisis de Imagen")
                col1, col2 = st.columns(2)
                
                with col1:
                    # Reducir el tamaño de la imagen original
                    resized_original = image_pil.resize((500, 500))
                    st.image(resized_original, caption="Imagen Original", use_column_width=False)
                
                with col2:
                    # Reducir el tamaño de la imagen preprocesada
                    img_pre_pil = Image.fromarray(img_pre)
                    resized_pre = img_pre_pil.resize((500, 500))
                    st.image(resized_pre, caption="Imagen Preprocesada", use_column_width=False)
                
                # Segunda fila: Gráficos
                st.subheader("Resultados del Análisis")
                col3, col4 = st.columns(2)
                
                with col3:
                    binary_chart = create_binary_probability_chart(predicted_proba)
                    st.plotly_chart(binary_chart, use_container_width=True)
                
                with col4:
                    multiclass_chart = create_multiclass_probability_chart(predicted_proba_multiclass)
                    st.plotly_chart(multiclass_chart, use_container_width=True)

                if st.button("📄 Generar Reporte PDF"):
                    pdf_buffer = create_pdf(
                        image_pil,
                        img_pre,
                        predicted_proba,
                        predicted_proba_multiclass,
                        predicted_class
                    )
                    
                    # Convertir PDF a base64 para descarga
                    b64_pdf = base64.b64encode(pdf_buffer.getvalue()).decode()
                    href = f'<a href="data:application/pdf;base64,{b64_pdf}" download="reporte_retinopatia.pdf">📥 Descargar Reporte PDF</a>'
                    st.markdown(href, unsafe_allow_html=True)
                
                # Resultado y recomendación
                if predicted_class == 0:
                    st.success(f"✅ El paciente en esta imagen se muestra sano de Retinopatía Diabética.")
                else:
                    severidad = {
                        1: "Leve",
                        2: "Moderada",
                        3: "Severa",
                        4: "Proliferativa"
                    }
                    nivel_severidad = severidad.get(predicted_multiclass[0], "No determinada")
                    st.warning(f"""
                    ⚠️ Se detectaron signos de Retinopatía Diabética {nivel_severidad}.
                    Se recomienda consultar con un especialista para una evaluación detallada.
                    """)
            else:
                st.error("No se pudo procesar la imagen correctamente. Por favor, intente con otra imagen.")
        else:
            st.error("Error: No se pudo cargar el modelo.")
            
    except Exception as e:
        st.error(f"Error al procesar la imagen: {str(e)}")
        st.info("Por favor, intente con otra imagen o verifique que la imagen sea válida.")

# Disclaimer y logos
st.markdown("---")
# Footer mejorado
st.markdown("""
    <div class="footer-container">
        <p style="font-weight: 500; color: #2c3e50;">
            ⚠️ IMPORTANTE: Este es un sistema de apoyo al diagnóstico.
        </p>
        <p style="color: #6c757d;">
            Las predicciones son estimaciones y no reemplazan el diagnóstico profesional.<br>
            Siempre consulte a un especialista en oftalmología. Este es un diagnóstico preliminar.
        </p>

    </div>
""", unsafe_allow_html=True)

# Logos en el pie de página
col1, col2 = st.columns(2)
with col1:
    st.image("./static/img/uvg.png", width=100)
with col2:
    st.image("./static/img/ceia.png", width=300)
