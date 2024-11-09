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
from reportlab.lib.utils import ImageReader

# Configuración de la página
st.set_page_config(
    page_title="Predictor de Retinopatía Diabética",
    layout="wide"
)
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
        binary_predictions = np.where(predicted_class == 0, 0, 1)[0]
        y_pred_prob_binary = np.sum(predictions[:, 1:], axis=1)[0]

        return binary_predictions, y_pred_prob_binary, predicted_class, predictions, preprocessed_img

    except Exception as e:
        st.error(f"Error en la predicción: {str(e)}")
        return None, None, None, None, None

def create_binary_probability_chart(prob):
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=['No Retinopatía', 'Retinopatía'],
        y=[1-prob, prob],
        marker_color=['#2ecc71', '#e74c3c']
    ))
    fig.update_layout(
        title='Probabilidad de Retinopatía Diabética',
        yaxis_title='Probabilidad',
        xaxis_title='Clase',
        showlegend=False,
        height=400
    )
    return fig

def create_multiclass_probability_chart(probs):
    labels = ['No RD', 'Leve', 'Moderada', 'Severa', 'Proliferativa']
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels,
        y=probs[0],
        marker_color=['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c', '#c0392b']
    ))
    fig.update_layout(
        title='Probabilidad por Nivel de Severidad',
        yaxis_title='Probabilidad',
        xaxis_title='Nivel de Severidad',
        showlegend=False,
        height=400
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

# Interfaz principal
#st.title("🏥 Predictor de Retinopatía Diabética")
#VERSION_MODELO = "1.0.0"
#st.write(f"Versión del modelo: {VERSION_MODELO}")



# Estilo CSS personalizado
st.markdown("""
    <style>
        .header-container {
            padding: 1.5rem;
            border-radius: 0.5rem;
            margin-bottom: 2rem;
            background-color: #f8f9fa;
        }
        .stButton > button {
            width: 300px !important;  /* Botones más anchos */
            height: 50px !important;
            background-color: #ffffff;
            border: 1px solid #ddd;
            border-radius: 5px;
            margin: 0 auto;
            display: block;
            text-align: center;
        }
        .stButton > button:hover {
            border-color: #4CAF50;
            color: #4CAF50;
        }
        .version-tag {
            color: #666;
            font-size: 0.8rem;
        }
        .custom-dev-text {
            text-align: center;
            padding: 0.5rem;
            margin-bottom: 1rem;
        }
        div[data-testid="column"] {
            display: flex;
            justify-content: center;
            align-items: center;
            flex-direction: column;
        }
    </style>
""", unsafe_allow_html=True)

# Título principal
st.markdown("""
    <div class="header-container">
        <h1 style='text-align: center; color: #2c3e50;'>🏥 Predictor de Retinopatía Diabética</h1>
        <p style='text-align: center;' class="version-tag">Versión del modelo: 1.0.0</p>
    </div>
""", unsafe_allow_html=True)
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    st.markdown('<div style="height: 20px;"></div>', unsafe_allow_html=True)
    if st.button("ℹ️ Información del Modelo"):
        st.info("""
        **Modelo de Deep Learning para Detección de Retinopatía**
        
        • Entrenado con miles de imágenes
        • Validado por oftalmólogos expertos
        • Actualización continua del modelo
        • Precisión superior al 90%
        """)

with col2:
    # Texto "Desarrollado por" centrado
    st.markdown('<p class="custom-dev-text">Desarrollado por:<br><strong>Juan Andrés Galicia Reyes</strong></p>', unsafe_allow_html=True)
    if st.button("📄 Licencia"):
        st.info("""
        **Licencia MIT**
        
        • Uso libre para investigación
        • Aplicaciones clínicas permitidas
        • Requiere reconocimiento
        • Sin garantía implícita
        """)

with col3:
    st.markdown('<div style="height: 20px;"></div>', unsafe_allow_html=True)
    if st.button("🔍 Actualizaciones"):
        st.info("""
        **Próximas Mejoras**
        
        • Mayor precisión en detección
        • Nuevos niveles de severidad
        • Mejoras en la interfaz
        • Análisis temporal de progresión
        """)

# Línea divisoria sutil
st.markdown("<hr style='margin: 2rem 0; border: none; border-top: 1px solid #eee;'>", unsafe_allow_html=True)


# Uploader de imagen
uploaded_file = st.file_uploader("Seleccione una imagen de fondo de ojo", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        modelo = load_keras_model()
        
        # Procesar imagen
        image_pil = Image.open(uploaded_file)
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
                    st.warning(f"""
                    ⚠️ Se detectaron signos de Retinopatía Diabética.
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
st.caption("""
⚠️ IMPORTANTE: Este es un sistema de apoyo al diagnóstico. 
Las predicciones son estimaciones y no reemplazan el diagnóstico profesional. 
Siempre consulte a un especialista en oftalmología.
""")

# Logos en el pie de página
col1, col2, col3 = st.columns(3)
with col1:
    st.image("./static/img/uvg.png", width=100)
with col2:
    st.image("./static/img/ceia.png", width=300)
#with col3:
    #st.image("./static/img/uno.png", width=150)