import streamlit as st
import pandas as pd
import joblib
from datetime import datetime
import logging
import requests
from io import BytesIO
import numpy as np

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Configuración de la página
st.set_page_config(
    page_title="Clasificador de Personalidad",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Aplicar estilo personalizado
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    .extrovert-result {
        background-color: #ff6b35;
        padding: 1.5rem;
        border-radius: 0.5rem;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .introvert-result {
        background-color: #4ecdc4;
        padding: 1.5rem;
        border-radius: 0.5rem;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metrics-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .debug-info {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border: 1px solid #dee2e6;
    }
    .personality-header {
        text-align: center;
        padding: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 1rem;
        margin-bottom: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def cargar_modelo():
    """Carga el modelo desde Google Drive o archivo local"""
    try:
        st.info("📤 Por favor, sube el archivo del modelo entrenado (.joblib)")
        uploaded_file = st.file_uploader("Selecciona el archivo del modelo", type=['joblib'])
        
        if uploaded_file is None:
            st.warning("⚠️ Necesitas subir un archivo de modelo para continuar")
            return None
        
        modelo_components = joblib.load(uploaded_file)
        
        # Verificar componentes requeridos
        required_components = ['modelo', 'scaler', 'encoders', 'selected_features', 'target_encoder']
        for component in required_components:
            if component not in modelo_components:
                st.error(f"❌ Componente faltante en el modelo: {component}")
                return None
        
        # Debug: Información del modelo
        with st.expander("🔍 Debug: Información del Modelo"):
            st.write("Características requeridas:", modelo_components['selected_features'])
            st.write("Columnas con encoders:", list(modelo_components['encoders'].keys()))
            st.write("Clases objetivo:", modelo_components['target_encoder'].classes_)
            if 'accuracy' in modelo_components:
                st.write("Accuracy del modelo:", f"{modelo_components['accuracy']:.4f}")
        
        st.success("✅ Modelo cargado exitosamente")
        return modelo_components
        
    except Exception as e:
        st.error(f"❌ Error al cargar el modelo: {str(e)}")
        with st.expander("🔍 Debug: Error Detallado"):
            st.write("Tipo de error:", type(e).__name__)
            st.write("Mensaje:", str(e))
        return None

def crear_campos_formulario():
    """Crea los campos del formulario basados en características comunes de personalidad"""
    st.markdown("""
        <div class="personality-header">
            <h2>🧠 Evaluador de Personalidad</h2>
            <p>Responde las siguientes preguntas para determinar tu tipo de personalidad</p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🗣️ Interacción Social")
        
        social_preference = st.selectbox(
            "¿Prefieres actividades sociales o individuales?",
            ["Actividades grupales", "Actividades individuales", "Depende del día"],
            help="Selecciona tu preferencia general"
        )
        
        communication_style = st.selectbox(
            "¿Cómo prefieres comunicarte?",
            ["Conversaciones largas", "Conversaciones cortas", "Por escrito", "Gestos/acciones"],
            help="Tu estilo de comunicación preferido"
        )
        
        energy_source = st.selectbox(
            "¿De dónde obtienes energía?",
            ["Estar con gente", "Estar solo", "Ambos por igual"],
            help="Lo que te recarga de energía"
        )
        
        party_behavior = st.selectbox(
            "En una fiesta, tú:",
            ["Hablas con muchas personas", "Hablas con pocas personas conocidas", 
             "Te quedas en un rincón", "Te vas temprano"],
            help="Tu comportamiento típico en fiestas"
        )

    with col2:
        st.markdown("#### 🤔 Procesamiento y Decisiones")
        
        decision_making = st.selectbox(
            "¿Cómo tomas decisiones?",
            ["Rápidamente", "Después de mucho pensar", "Consultando con otros", "Por impulso"],
            help="Tu proceso de toma de decisiones"
        )
        
        thinking_style = st.selectbox(
            "¿Cómo prefieres procesar información?",
            ["Pensando en voz alta", "Reflexionando internamente", 
             "Discutiendo con otros", "Escribiendo"],
            help="Tu estilo de procesamiento mental"
        )
        
        stress_response = st.selectbox(
            "Cuando estás estresado, prefieres:",
            ["Hablar con alguien", "Estar solo", "Hacer ejercicio", "Dormir"],
            help="Tu respuesta típica al estrés"
        )
        
        weekend_preference = st.selectbox(
            "Un fin de semana ideal incluye:",
            ["Salir con amigos", "Quedarse en casa", "Actividades al aire libre", 
             "Leer o ver películas"],
            help="Tu tipo de fin de semana preferido"
        )
    
    # Campos adicionales basados en escalas Likert
    st.markdown("#### 📊 Evaluación por Escalas")
    
    col3, col4 = st.columns(2)
    
    with col3:
        sociability = st.slider(
            "Nivel de sociabilidad (1=muy bajo, 10=muy alto)",
            min_value=1, max_value=10, value=5,
            help="Qué tan sociable te consideras"
        )
        
        assertiveness = st.slider(
            "Nivel de asertividad (1=muy pasivo, 10=muy asertivo)",
            min_value=1, max_value=10, value=5,
            help="Qué tan asertivo eres en las interacciones"
        )
    
    with col4:
        activity_level = st.slider(
            "Nivel de actividad (1=muy tranquilo, 10=muy activo)",
            min_value=1, max_value=10, value=5,
            help="Tu nivel general de actividad"
        )
        
        emotional_expression = st.slider(
            "Expresión emocional (1=muy reservado, 10=muy expresivo)",
            min_value=1, max_value=10, value=5,
            help="Qué tan expresivo eres emocionalmente"
        )

    return {
        'social_preference': social_preference,
        'communication_style': communication_style,
        'energy_source': energy_source,
        'party_behavior': party_behavior,
        'decision_making': decision_making,
        'thinking_style': thinking_style,
        'stress_response': stress_response,
        'weekend_preference': weekend_preference,
        'sociability': sociability,
        'assertiveness': assertiveness,
        'activity_level': activity_level,
        'emotional_expression': emotional_expression
    }

def preparar_datos_para_modelo(datos, selected_features, encoders):
    """Prepara los datos del formulario según las características requeridas por el modelo"""
    
    # Crear un mapeo de las respuestas a valores numéricos
    feature_mapping = {
        'social_preference': {
            'Actividades grupales': 2, 'Depende del día': 1, 'Actividades individuales': 0
        },
        'communication_style': {
            'Conversaciones largas': 3, 'Conversaciones cortas': 1, 'Por escrito': 0, 'Gestos/acciones': 2
        },
        'energy_source': {
            'Estar con gente': 2, 'Ambos por igual': 1, 'Estar solo': 0
        },
        'party_behavior': {
            'Hablas con muchas personas': 3, 'Hablas con pocas personas conocidas': 2,
            'Te quedas en un rincón': 1, 'Te vas temprano': 0
        },
        'decision_making': {
            'Por impulso': 3, 'Rápidamente': 2, 'Consultando con otros': 1, 'Después de mucho pensar': 0
        },
        'thinking_style': {
            'Pensando en voz alta': 3, 'Discutiendo con otros': 2, 'Escribiendo': 1, 'Reflexionando internamente': 0
        },
        'stress_response': {
            'Hablar con alguien': 2, 'Hacer ejercicio': 1, 'Dormir': 0, 'Estar solo': 0
        },
        'weekend_preference': {
            'Salir con amigos': 3, 'Actividades al aire libre': 2, 'Leer o ver películas': 0, 'Quedarse en casa': 0
        }
    }
    
    # Crear DataFrame base con todas las características posibles
    df_base = pd.DataFrame({
        'social_preference': [feature_mapping['social_preference'].get(datos['social_preference'], 1)],
        'communication_style': [feature_mapping['communication_style'].get(datos['communication_style'], 1)],
        'energy_source': [feature_mapping['energy_source'].get(datos['energy_source'], 1)],
        'party_behavior': [feature_mapping['party_behavior'].get(datos['party_behavior'], 1)],
        'decision_making': [feature_mapping['decision_making'].get(datos['decision_making'], 1)],
        'thinking_style': [feature_mapping['thinking_style'].get(datos['thinking_style'], 1)],
        'stress_response': [feature_mapping['stress_response'].get(datos['stress_response'], 1)],
        'weekend_preference': [feature_mapping['weekend_preference'].get(datos['weekend_preference'], 1)],
        'sociability': [datos['sociability']],
        'assertiveness': [datos['assertiveness']],
        'activity_level': [datos['activity_level']],
        'emotional_expression': [datos['emotional_expression']]
    })
    
    # Agregar características derivadas comunes
    df_base['social_score'] = (df_base['sociability'] + df_base['assertiveness']) / 2
    df_base['activity_expression'] = df_base['activity_level'] * df_base['emotional_expression'] / 10
    df_base['extroversion_indicator'] = (
        df_base['social_preference'] + df_base['energy_source'] + df_base['party_behavior']
    ) / 3
    
    # Seleccionar solo las características que el modelo necesita
    available_features = [col for col in selected_features if col in df_base.columns]
    
    if not available_features:
        available_features = df_base.columns.tolist()
        st.warning("⚠️ Las características del formulario no coinciden exactamente con el modelo. Usando mapeo genérico.")
    
    # Si faltan características, agregar valores por defecto
    for feature in selected_features:
        if feature not in df_base.columns:
            df_base[feature] = 0
    
    # Debug: Mostrar información de preparación
    with st.expander("🔍 Debug: Preparación de Datos"):
        st.write("1. Características requeridas por el modelo:", selected_features)
        st.write("2. Características disponibles:", available_features)
        st.write("3. DataFrame preparado:")
        st.dataframe(df_base[selected_features])
        
        missing_features = [col for col in selected_features if col not in df_base.columns]
        if missing_features:
            st.warning(f"⚠️ Características faltantes (usando valores por defecto): {missing_features}")
    
    return df_base[selected_features]

def procesar_prediccion(df_preparado, modelo_components):
    """Procesa la predicción con el modelo"""
    try:
        modelo = modelo_components['modelo']
        scaler = modelo_components['scaler']
        target_encoder = modelo_components['target_encoder']
        
        # Aplicar encoders a variables categóricas si es necesario
        df_encoded = df_preparado.copy()
        
        # Debug: Mostrar datos antes del scaling
        with st.expander("🔍 Debug: Datos Antes del Scaling"):
            st.write("Valores antes de escalar:", df_encoded.iloc[0].to_dict())
        
        # Asegurar que todas las columnas son numéricas
        for col in df_encoded.columns:
            df_encoded[col] = pd.to_numeric(df_encoded[col], errors='coerce')
        
        # Rellenar cualquier NaN con 0
        df_encoded.fillna(0, inplace=True)
        
        # Escalar datos
        datos_scaled = scaler.transform(df_encoded)
        
        # Debug: Mostrar datos después del scaling
        with st.expander("🔍 Debug: Datos Después del Scaling"):
            st.write("Shape de datos escalados:", datos_scaled.shape)
            st.write("Datos escalados:", datos_scaled[0])
        
        # Realizar predicción
        prediccion = modelo.predict(datos_scaled)[0]
        
        # Obtener probabilidades si está disponible
        probabilidades = None
        try:
            if hasattr(modelo, 'decision_function'):
                decision_scores = modelo.decision_function(datos_scaled)[0]
                prob_extrovert = 1 / (1 + np.exp(-decision_scores))
                probabilidades = [1 - prob_extrovert, prob_extrovert]
        except:
            pass
        
        # Decodificar la predicción
        personalidad = target_encoder.inverse_transform([prediccion])[0]
        
        return personalidad, probabilidades
        
    except Exception as e:
        st.error(f"❌ Error en el procesamiento: {str(e)}")
        with st.expander("🔍 Debug: Error Detallado"):
            st.write("Tipo de error:", type(e).__name__)
            st.write("Mensaje:", str(e))
        return None, None

def mostrar_resultado(personalidad, datos, probabilidades=None):
    """Muestra el resultado de la predicción de personalidad"""
    st.header("🎭 Resultado del Análisis de Personalidad")
    
    # Determinar si es introvertido o extrovertido
    es_extrovertido = personalidad.lower() in ['extrovert', 'extrovertido', 'e', '1']
    
    if es_extrovertido:
        st.markdown("""
            <div class="extrovert-result">
                <h3>🌟 ¡PERSONALIDAD EXTROVERTIDA!</h3>
                <p><strong>Características principales:</strong></p>
                <ul>
                    <li>Te energizas estando con otras personas</li>
                    <li>Prefieres procesar información hablando</li>
                    <li>Eres expresivo y sociable</li>
                    <li>Te sientes cómodo siendo el centro de atención</li>
                    <li>Tomas decisiones rápidamente</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div class="introvert-result">
                <h3>🌙 ¡PERSONALIDAD INTROVERTIDA!</h3>
                <p><strong>Características principales:</strong></p>
                <ul>
                    <li>Te energizas estando solo o con pocas personas</li>
                    <li>Prefieres procesar información internamente</li>
                    <li>Eres reflexivo y observador</li>
                    <li>Prefieres conversaciones profundas</li>
                    <li>Piensas antes de hablar o actuar</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

    # Mostrar métricas
    st.markdown('<div class="metrics-container">', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Tipo de Personalidad",
            value="Extrovertido" if es_extrovertido else "Introvertido"
        )
    
    with col2:
        st.metric(
            label="Nivel de Sociabilidad",
            value=f"{datos['sociability']}/10"
        )
    
    with col3:
        st.metric(
            label="Nivel de Actividad",
            value=f"{datos['activity_level']}/10"
        )
    
    with col4:
        if probabilidades is not None:
            confianza = max(probabilidades) if probabilidades else 0.5
            st.metric(
                label="Confianza",
                value=f"{confianza:.1%}"
            )
        else:
            st.metric(
                label="Confianza",
                value="N/A"
            )
    st.markdown('</div>', unsafe_allow_html=True)

    # Interpretación personalizada
    st.markdown("### 📊 Interpretación Detallada")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🎯 Fortalezas Identificadas")
        if es_extrovertido:
            fortalezas = [
                "Habilidades de comunicación",
                "Liderazgo natural",
                "Adaptabilidad social",
                "Energía contagiosa",
                "Facilidad para hacer networking"
            ]
        else:
            fortalezas = [
                "Capacidad de análisis profundo",
                "Escucha activa",
                "Creatividad e innovación",
                "Concentración sostenida",
                "Relaciones significativas"
            ]
        
        for fortaleza in fortalezas:
            st.write(f"• {fortaleza}")
    
    with col2:
        st.markdown("#### 💡 Recomendaciones")
        if es_extrovertido:
            recomendaciones = [
                "Practica la escucha activa",
                "Toma tiempo para reflexionar antes de decidir",
                "Desarrolla relaciones más profundas",
                "Aprovecha tu energía para motivar equipos",
                "Busca roles que involucren interacción social"
            ]
        else:
            recomendaciones = [
                "Practica hablar en público gradualmente",
                "Aprovecha tu capacidad de observación",
                "Busca ambientes de trabajo tranquilos",
                "Desarrolla tu red de contactos de forma selectiva",
                "Valora tus momentos de soledad para recargar"
            ]
        
        for recomendacion in recomendaciones:
            st.write(f"• {recomendacion}")

    # Mostrar detalles de la evaluación
    with st.expander("📝 Detalles de la Evaluación"):
        resultado_detallado = {
            'Tipo de Personalidad': "Extrovertido" if es_extrovertido else "Introvertido",
            'Preferencia Social': datos['social_preference'],
            'Fuente de Energía': datos['energy_source'],
            'Estilo de Comunicación': datos['communication_style'],
            'Comportamiento en Fiestas': datos['party_behavior'],
            'Estilo de Toma de Decisiones': datos['decision_making'],
            'Respuesta al Estrés': datos['stress_response'],
            'Puntuaciones': {
                'Sociabilidad': f"{datos['sociability']}/10",
                'Asertividad': f"{datos['assertiveness']}/10", 
                'Nivel de Actividad': f"{datos['activity_level']}/10",
                'Expresión Emocional': f"{datos['emotional_expression']}/10"
            }
        }
        
        if probabilidades:
            resultado_detallado['Probabilidades'] = {
                'Introvertido': f"{probabilidades[0]:.1%}",
                'Extrovertido': f"{probabilidades[1]:.1%}"
            }
        
        st.json(resultado_detallado)
        st.write(f"🕒 Evaluación realizada el: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def validar_datos_entrada(datos):
    """Valida los datos de entrada del formulario"""
    errores = []
    
    # Validar que se hayan seleccionado todas las opciones
    campos_requeridos = [
        'social_preference', 'communication_style', 'energy_source', 
        'party_behavior', 'decision_making', 'thinking_style', 
        'stress_response', 'weekend_preference'
    ]
    
    for campo in campos_requeridos:
        if not datos.get(campo) or datos[campo] == "":
            errores.append(f"Debe seleccionar una opción para: {campo.replace('_', ' ').title()}")
    
    # Validar rangos de escalas
    escalas = ['sociability', 'assertiveness', 'activity_level', 'emotional_expression']
    for escala in escalas:
        if datos[escala] < 1 or datos[escala] > 10:
            errores.append(f"El valor de {escala.replace('_', ' ')} debe estar entre 1 y 10")
    
    return errores

def main():
    """Función principal de la aplicación Streamlit"""
    st.title("🧠 Clasificador de Personalidad: Introvertido vs Extrovertido")
    
    # Información sobre la aplicación
    with st.sidebar:
        st.markdown("""
        ### 📋 Acerca de esta App
        
        Esta aplicación utiliza machine learning para clasificar tu personalidad como **Introvertida** o **Extrovertida** basándose en tus respuestas a un cuestionario.
        
        #### 🔬 Cómo funciona:
        1. **Respondes** a preguntas sobre tus preferencias
        2. **El modelo** analiza tus respuestas  
        3. **Obtienes** tu clasificación de personalidad
        
        #### 📊 Características evaluadas:
        - Preferencias sociales
        - Estilos de comunicación
        - Fuentes de energía
        - Procesamiento de información
        - Comportamiento en grupos
        """)
        
        st.markdown("---")
        st.markdown("🎯 **Precisión del modelo**: Basado en características validadas de personalidad")

    # Cargar el modelo
    modelo_components = cargar_modelo()
    if modelo_components is None:
        st.stop()

    # Crear formulario
    with st.form("personality_form"):
        st.markdown("### 📝 Cuestionario de Personalidad")
        st.markdown("*Responde honestamente a las siguientes preguntas. No hay respuestas correctas o incorrectas.*")
        
        datos = crear_campos_formulario()
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            submitted = st.form_submit_button("🧠 Analizar mi Personalidad", use_container_width=True)

    if submitted:
        # Validar datos
        errores = validar_datos_entrada(datos)
        if errores:
            st.error("❌ Por favor corrige los siguientes errores:")
            for error in errores:
                st.write(f"• {error}")
            return

        try:
            with st.spinner("🔍 Analizando tu personalidad..."):
                # Preparar datos
                df_preparado = preparar_datos_para_modelo(
                    datos, 
                    modelo_components['selected_features'],
                    modelo_components['encoders']
                )
                
                # Procesar predicción
                personalidad, probabilidades = procesar_prediccion(
                    df_preparado, 
                    modelo_components
                )
                
                if personalidad is not None:
                    # Mostrar resultado
                    mostrar_resultado(personalidad, datos, probabilidades)
                    
                    # Logging
                    logging.info(f"Predicción realizada: {personalidad}")
                    
                    # Botón para nueva evaluación
                    if st.button("🔄 Realizar nueva evaluación"):
                        st.rerun()
                
        except Exception as e:
            st.error(f"❌ Error al procesar la evaluación: {str(e)}")
            with st.expander("🔍 Debug: Error Detallado"):
                st.write("Tipo de error:", type(e).__name__)
                st.write("Mensaje:", str(e))

    # Información adicional
    with st.expander("ℹ️ Información sobre Introversión y Extroversión"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🌙 Características Introvertidas
            - Se energizan con la soledad
            - Prefieren conversaciones profundas
            - Procesan información internamente
            - Piensan antes de hablar
            - Prefieren pocos amigos cercanos
            - Necesitan tiempo para recargar después de socializar
            """)
        
        with col2:
            st.markdown("""
            ### 🌟 Características Extrovertidas  
            - Se energizan con la interacción social
            - Disfrutan conversaciones grupales
            - Procesan información externamente
            - Piensan mientras hablan
            - Tienen muchos conocidos
            - Se recargan estando con otros
            """)
        
        st.markdown("""
        ### 🎯 Nota Importante
        La introversión y extroversión son un espectro, no categorías rígidas. Muchas personas muestran características de ambos tipos (ambivertidos). Esta evaluación proporciona una tendencia general basada en tus respuestas.
        """)

if __name__ == "__main__":
    main()
