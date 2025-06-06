import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
from io import BytesIO
import time

# Configuración de la página
st.set_page_config(
    page_title="🧠 Predictor de Personalidad",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .feature-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
    }
    .prediction-card {
        text-align: center;
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
    }
    .introvert-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    .extrovert-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
    }
    .stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def cargar_modelo():
    """Carga el modelo desde Google Drive"""
    try:
        # URL de Google Drive (reemplazar con tu ID)
        GDRIVE_FILE_ID = "1TybTQh4Pt4tFfNq9lOwzTZTHBMnnfVxm"
        download_url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
        
        st.info("📥 Descargando modelo...")
        
        response = requests.get(download_url)
        if response.status_code != 200:
            st.error("❌ Error al descargar el modelo de Google Drive")
            return None
            
        model_bytes = BytesIO(response.content)
        modelo_components = joblib.load(model_bytes)
        
        st.success("✅ Modelo cargado exitosamente")
        return modelo_components
        
    except Exception as e:
        st.error(f"❌ Error al cargar el modelo: {str(e)}")
        return None

def procesar_entrada(data, modelo_components):
    """Procesa los datos de entrada y hace la predicción"""
    try:
        # Extraer componentes del modelo
        modelo = modelo_components['modelo']
        scaler = modelo_components['scaler']
        encoders = modelo_components['encoders']
        selected_features = modelo_components['selected_features']
        target_encoder = modelo_components['target_encoder']
        
        # Crear DataFrame con las características seleccionadas
        df_input = pd.DataFrame([data])
        
        # Procesar variables categóricas
        for col in ['Stage_fear', 'Drained_after_socializing']:
            if col in df_input.columns and col in encoders:
                # Convertir a string para asegurar compatibilidad
                df_input[col] = df_input[col].astype(str)
                df_input[col] = encoders[col].transform(df_input[col])
        
        # Seleccionar solo las características del modelo
        df_final = df_input[selected_features]
        
        # Estandarizar
        X_scaled = scaler.transform(df_final)
        
        # Predecir
        prediction = modelo.predict(X_scaled)[0]
        probability = modelo.decision_function(X_scaled)[0]
        
        # Convertir predicción a etiqueta
        resultado = target_encoder.inverse_transform([prediction])[0]
        
        # Calcular confianza (normalizar el decision_function a probabilidad aproximada)
        confianza = 1 / (1 + np.exp(-abs(probability)))
        
        return resultado, confianza
        
    except Exception as e:
        st.error(f"❌ Error en el procesamiento: {str(e)}")
        return None, None

def main():
    # Header principal
    st.markdown("""
    <div class="main-header">
        <h1>🧠 Predictor de Personalidad</h1>
        <p>Descubre si eres más Introvertido o Extrovertido</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Cargar modelo
    modelo_components = cargar_modelo()
    
    if modelo_components is None:
        st.error("❌ No se pudo cargar el modelo. Verifica la configuración.")
        st.stop()
    
    # Información del modelo
    with st.expander("ℹ️ Información del Modelo"):
        accuracy = modelo_components.get('accuracy', 0)
        st.write(f"**Precisión del modelo:** {accuracy:.2%}")
        st.write(f"**Algoritmo:** {modelo_components.get('model_info', {}).get('algorithm', 'LinearSVC')}")
        st.write(f"**Características utilizadas:** {len(modelo_components['selected_features'])}")
        st.write("**Clases predichas:** Introvertido, Extrovertido")
    
    st.markdown("### 📋 Completa el siguiente cuestionario:")
    
    # Formulario principal
    with st.form("personality_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="feature-card">', unsafe_allow_html=True)
            st.markdown("**⏰ Tiempo Social**")
            time_alone = st.slider(
                "¿Cuántas horas al día prefieres estar solo?",
                min_value=0, max_value=12, value=4,
                help="Indica tu preferencia de tiempo a solas"
            )
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="feature-card">', unsafe_allow_html=True)
            st.markdown("**👥 Vida Social**")
            social_events = st.slider(
                "¿Cuántos eventos sociales asistes al mes?",
                min_value=0, max_value=20, value=5,
                help="Fiestas, reuniones, eventos, etc."
            )
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="feature-card">', unsafe_allow_html=True)
            st.markdown("**🚶 Actividades Externas**")
            going_outside = st.slider(
                "¿Con qué frecuencia sales de casa? (días por semana)",
                min_value=0, max_value=7, value=4,
                help="Para actividades no laborales"
            )
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="feature-card">', unsafe_allow_html=True)
            st.markdown("**📱 Actividad en Redes**")
            post_frequency = st.slider(
                "¿Cuántas publicaciones haces por semana en redes sociales?",
                min_value=0, max_value=50, value=5
            )
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="feature-card">', unsafe_allow_html=True)
            st.markdown("**🎭 Miedo Escénico**")
            stage_fear = st.radio(
                "¿Tienes miedo a hablar en público?",
                options=["No", "Yes"],
                format_func=lambda x: "No" if x == "No" else "Sí"
            )
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="feature-card">', unsafe_allow_html=True)
            st.markdown("**😴 Energía Social**")
            drained_socializing = st.radio(
                "¿Te sientes agotado después de socializar?",
                options=["No", "Yes"],
                format_func=lambda x: "No" if x == "No" else "Sí"
            )
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="feature-card">', unsafe_allow_html=True)
            st.markdown("**👫 Círculo Social**")
            friends_circle = st.slider(
                "¿Cuántos amigos cercanos tienes?",
                min_value=0, max_value=50, value=8,
                help="Personas con las que tienes contacto regular"
            )
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Botón de predicción
        submitted = st.form_submit_button("🔮 Predecir mi Personalidad")
    
    # Procesar predicción
    if submitted:
        # Preparar datos
        input_data = {
            'Time_spent_Alone': time_alone,
            'Stage_fear': stage_fear,
            'Social_event_attendance': social_events,
            'Going_outside': going_outside,
            'Drained_after_socializing': drained_socializing,
            'Friends_circle_size': friends_circle,
            'Post_frequency': post_frequency
        }
        
        # Mostrar loading
        with st.spinner('🧠 Analizando tu personalidad...'):
            time.sleep(1)  # Efecto dramático
            resultado, confianza = procesar_entrada(input_data, modelo_components)
        
        if resultado:
            st.markdown("---")
            
            # Mostrar resultado
            if resultado == "Introvert":
                st.markdown("""
                <div class="prediction-card introvert-card">
                    <h2>🧘 Eres más INTROVERTIDO</h2>
                    <p>Prefieres la reflexión interna y ambientes más tranquilos</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("""
                ### 🔍 Características de tu personalidad:
                - 🤔 **Reflexivo:** Prefieres pensar antes de actuar
                - 🏠 **Hogareño:** Disfrutas del tiempo en espacios familiares
                - 👥 **Círculo pequeño:** Prefieres pocos amigos pero cercanos
                - 🎯 **Enfocado:** Te concentras bien en tareas individuales
                - 📚 **Observador:** Aprendes mejor observando que participando
                """)
                
            else:  # Extrovert
                st.markdown("""
                <div class="prediction-card extrovert-card">
                    <h2>🎉 Eres más EXTROVERTIDO</h2>
                    <p>Te energizas con la interacción social y actividades grupales</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("""
                ### 🔍 Características de tu personalidad:
                - 🗣️ **Comunicativo:** Disfrutas expresar tus ideas verbalmente
                - 👥 **Social:** Te sientes cómodo en grupos grandes
                - ⚡ **Energético:** La interacción social te da energía
                - 🎯 **Espontáneo:** Prefieres decidir sobre la marcha
                - 🤝 **Colaborativo:** Trabajas mejor en equipo
                """)
            
            # Mostrar confianza
            st.markdown(f"""
            <div style="text-align: center; margin: 1rem 0;">
                <h4>Nivel de confianza: {confianza:.1%}</h4>
                <div style="background: #e0e0e0; border-radius: 10px; overflow: hidden;">
                    <div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
                                height: 20px; width: {confianza*100:.1f}%; 
                                transition: width 0.5s ease;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Disclaimer
            st.markdown("""
            ---
            **💡 Nota importante:** Esta predicción se basa en un modelo de aprendizaje automático 
            y tiene fines educativos. La personalidad es compleja y multifacética. 
            Los resultados deben interpretarse como una orientación general, no como un diagnóstico definitivo.
            """)
            
            # Botón para nueva predicción
            if st.button("🔄 Hacer nueva predicción"):
                st.rerun()

if __name__ == "__main__":
    main()
