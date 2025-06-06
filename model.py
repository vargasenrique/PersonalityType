# personality_model_trainer.py
# Script consolidado para entrenar y guardar modelo de personalidad

# Instalación de dependencias
!pip install kaggle seaborn scikit-learn

import os
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from google.colab import files

# Sklearn imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif

# Visualization imports
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración inicial
sns.set_theme()
pd.set_option('display.max_columns', None)

class PersonalityModelTrainer:
    """Clase principal para entrenar y guardar el modelo de personalidad"""
    
    def __init__(self):
        self.df = None
        self.target_col = None
        self.modelo = None
        self.scaler = None
        self.encoders = {}
        self.selected_features = []
        self.target_encoder = None
        self.accuracy = 0.0
        
    def setup_kaggle(self):
        """Configura el entorno de Kaggle en Colab"""
        print("🔧 CONFIGURANDO KAGGLE")
        print("=" * 50)
        
        !mkdir -p ~/.kaggle
        print("Por favor, sube tu archivo kaggle.json:")
        uploaded = files.upload()
        !cp kaggle.json ~/.kaggle/
        !chmod 600 ~/.kaggle/kaggle.json
        print("✅ Configuración de Kaggle completada\n")
    
    def load_data(self):
        """Descarga y carga el dataset de personalidad"""
        print("📥 CARGANDO DATASET")
        print("=" * 50)
        
        !kaggle datasets download -d rakeshkapilavai/extrovert-vs-introvert-behavior-data --unzip
        self.df = pd.read_csv('personality_dataset.csv')
        print(f"✅ Dataset cargado: {self.df.shape[0]} filas y {self.df.shape[1]} columnas\n")
        
    def analisis_exploratorio(self):
        """Realiza análisis exploratorio y selecciona columna objetivo"""
        print("🔍 ANÁLISIS EXPLORATORIO")
        print("=" * 50)
        
        # Información básica
        print(f"📊 Registros: {len(self.df)}")
        print(f"📊 Características: {self.df.shape[1]}")
        print(f"📊 Columnas: {list(self.df.columns)}")
        
        # Detectar columna objetivo automáticamente
        target_keywords = ['personality', 'type', 'label', 'class', 'extrovert', 'introvert']
        self.target_col = None
        
        for col in self.df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in target_keywords):
                self.target_col = col
                break
        
        # Si no se encuentra automáticamente, mostrar opciones
        if self.target_col is None:
            print("\n🎯 SELECCIÓN DE COLUMNA OBJETIVO:")
            for i, col in enumerate(self.df.columns):
                unique_vals = self.df[col].nunique()
                print(f"{i}: {col} (valores únicos: {unique_vals})")
                if unique_vals <= 10:
                    print(f"   Valores: {list(self.df[col].unique())}")
            
            col_index = int(input("\nIngrese el número de la columna objetivo: "))
            self.target_col = self.df.columns[col_index]
        
        print(f"✅ Columna objetivo: {self.target_col}")
        
        # Distribución de clases
        print(f"\n📈 DISTRIBUCIÓN DE CLASES:")
        dist = self.df[self.target_col].value_counts()
        for categoria, count in dist.items():
            print(f"  {categoria}: {count} ({(count/len(self.df))*100:.1f}%)")
        
        # Visualización
        plt.figure(figsize=(12, 8))
        
        # Distribución de clases
        plt.subplot(2, 2, 1)
        self.df[self.target_col].value_counts().plot(kind='bar', color='skyblue')
        plt.title('Distribución de Personalidades')
        plt.xlabel('Tipo de Personalidad')
        plt.ylabel('Cantidad')
        plt.xticks(rotation=45)
        
        # Matriz de correlación
        plt.subplot(2, 2, 2)
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_matrix = self.df[numeric_cols].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, cbar_kws={'shrink': 0.8})
            plt.title('Correlación entre Variables')
        
        # Valores faltantes
        plt.subplot(2, 2, 3)
        missing_data = self.df.isnull().sum()
        if missing_data.sum() > 0:
            missing_data[missing_data > 0].plot(kind='bar', color='salmon')
            plt.title('Valores Faltantes por Columna')
            plt.xticks(rotation=45)
        else:
            plt.text(0.5, 0.5, 'No hay valores\nfaltantes', ha='center', va='center', 
                    transform=plt.gca().transAxes, fontsize=12)
            plt.title('Valores Faltantes')
        
        # Estadísticas básicas
        plt.subplot(2, 2, 4)
        numeric_stats = self.df[numeric_cols].describe()
        plt.text(0.1, 0.9, f'Variables numéricas: {len(numeric_cols)}', 
                transform=plt.gca().transAxes, fontsize=10)
        plt.text(0.1, 0.8, f'Variables categóricas: {len(self.df.select_dtypes(include=[object]).columns)}', 
                transform=plt.gca().transAxes, fontsize=10)
        plt.text(0.1, 0.7, f'Clases objetivo: {self.df[self.target_col].nunique()}', 
                transform=plt.gca().transAxes, fontsize=10)
        plt.title('Resumen del Dataset')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        print("✅ Análisis exploratorio completado\n")
    
    def preparar_datos(self, n_features=15):
        """Prepara los datos para el entrenamiento"""
        print("🔄 PREPARACIÓN DE DATOS")
        print("=" * 50)
        
        df_prep = self.df.copy()
        
        # 1. Manejo de valores faltantes
        print("📋 Procesando valores faltantes...")
        numericas = df_prep.select_dtypes(include=['int64', 'float64']).columns
        categoricas = df_prep.select_dtypes(include=['object']).columns
        
        for col in numericas:
            if col != self.target_col:
                df_prep[col].fillna(df_prep[col].median(), inplace=True)
        
        for col in categoricas:
            if col != self.target_col:
                mode_val = df_prep[col].mode()
                fill_val = mode_val[0] if len(mode_val) > 0 else 'unknown'
                df_prep[col].fillna(fill_val, inplace=True)
        
        # 2. Codificación de variables categóricas
        print("🔤 Codificando variables categóricas...")
        self.encoders = {}
        for col in categoricas:
            if col != self.target_col:
                self.encoders[col] = LabelEncoder()
                df_prep[col] = self.encoders[col].fit_transform(df_prep[col].astype(str))
        
        # 3. Preparar variable objetivo
        print("🎯 Preparando variable objetivo...")
        self.target_encoder = LabelEncoder()
        y = self.target_encoder.fit_transform(df_prep[self.target_col])
        
        target_mapping = dict(zip(self.target_encoder.classes_, 
                                self.target_encoder.transform(self.target_encoder.classes_)))
        print(f"   Mapeo de clases: {target_mapping}")
        
        # 4. Separar features
        X = df_prep.drop(self.target_col, axis=1)
        
        # Asegurar que todas las features son numéricas
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')
        X.fillna(X.median(), inplace=True)
        
        print(f"   Shape de X: {X.shape}, Shape de y: {y.shape}")
        
        # 5. Selección de características
        if n_features < X.shape[1]:
            print(f"⚡ Seleccionando las {n_features} características más importantes...")
            selector = SelectKBest(f_classif, k=min(n_features, X.shape[1]))
            X_selected = selector.fit_transform(X, y)
            self.selected_features = X.columns[selector.get_support()].tolist()
        else:
            print("⚡ Usando todas las características disponibles...")
            X_selected = X.values
            self.selected_features = X.columns.tolist()
        
        print(f"   Características seleccionadas: {len(self.selected_features)}")
        
        # 6. División de datos
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # 7. Estandarización
        print("📏 Estandarizando datos...")
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print("✅ Preparación de datos completada\n")
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def entrenar_modelo(self, X_train, y_train, sample_size=0.8):
        """Entrena el modelo de clasificación"""
        print("🚀 ENTRENAMIENTO DEL MODELO")
        print("=" * 50)
        
        # Reducir dataset si es muy grande
        if len(X_train) > 10000:
            print(f"📊 Reduciendo dataset al {sample_size*100:.0f}% para velocidad...")
            X_train_reduced, _, y_train_reduced, _ = train_test_split(
                X_train, y_train, train_size=sample_size, stratify=y_train, random_state=42
            )
        else:
            X_train_reduced, y_train_reduced = X_train, y_train
        
        print(f"🎯 Entrenando con {len(X_train_reduced)} muestras...")
        
        # Entrenar modelo
        self.modelo = LinearSVC(
            random_state=42,
            dual=False,
            max_iter=2000,
            tol=1e-4,
            class_weight='balanced'
        )
        
        self.modelo.fit(X_train_reduced, y_train_reduced)
        print("✅ Modelo entrenado exitosamente\n")
    
    def evaluar_modelo(self, X_test, y_test):
        """Evalúa el rendimiento del modelo"""
        print("📊 EVALUACIÓN DEL MODELO")
        print("=" * 50)
        
        # Predicciones
        y_pred = self.modelo.predict(X_test)
        self.accuracy = accuracy_score(y_test, y_pred)
        
        print(f"🎯 Accuracy: {self.accuracy:.4f} ({self.accuracy*100:.2f}%)")
        
        # Reporte detallado
        print("\n📋 Reporte de Clasificación:")
        target_names = self.target_encoder.classes_
        print(classification_report(y_test, y_pred, target_names=target_names))
        
        # Matriz de confusión
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=target_names, yticklabels=target_names)
        plt.title(f'Matriz de Confusión - Accuracy: {self.accuracy:.2%}')
        plt.ylabel('Valor Real')
        plt.xlabel('Valor Predicho')
        plt.show()
        
        print("✅ Evaluación completada\n")
    
    def guardar_modelo_completo(self):
        """Guarda el modelo completo con toda la información necesaria"""
        print("💾 GUARDANDO MODELO COMPLETO")
        print("=" * 50)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Componentes del modelo
        modelo_components = {
            'modelo': self.modelo,
            'scaler': self.scaler,
            'encoders': self.encoders,
            'selected_features': self.selected_features,
            'target_encoder': self.target_encoder,
            'accuracy': self.accuracy,
            'timestamp': timestamp,
            'model_info': {
                'algorithm': 'LinearSVC',
                'features_count': len(self.selected_features),
                'target_classes': self.target_encoder.classes_.tolist(),
                'training_date': datetime.now().isoformat()
            }
        }
        
        # Guardar modelo principal
        filename = f'modelo_personalidad_{timestamp}.joblib'
        joblib.dump(modelo_components, filename)
        print(f"✅ Modelo guardado: {filename}")
        
        # Información de debug
        debug_info = {
            'model_metadata': {
                'algorithm': 'LinearSVC',
                'accuracy': float(self.accuracy),
                'timestamp': timestamp,
                'training_date': datetime.now().isoformat()
            },
            'features': {
                'selected_features': self.selected_features,
                'features_count': len(self.selected_features)
            },
            'encoders': {
                'categorical_columns': list(self.encoders.keys()),
                'encoder_classes': {col: encoder.classes_.tolist() 
                                 for col, encoder in self.encoders.items()}
            },
            'target_encoder': {
                'classes': self.target_encoder.classes_.tolist(),
                'class_mapping': dict(zip(self.target_encoder.classes_, 
                                        range(len(self.target_encoder.classes_))))
            }
        }
        
        debug_filename = f'modelo_debug_{timestamp}.json'
        with open(debug_filename, 'w', encoding='utf-8') as f:
            json.dump(debug_info, f, indent=4, ensure_ascii=False)
        print(f"✅ Debug info guardada: {debug_filename}")
        
        # README
        readme_content = f"""# Modelo de Personalidad - {timestamp}

## Información del Modelo
- **Algoritmo**: LinearSVC
- **Accuracy**: {self.accuracy:.4f} ({self.accuracy*100:.2f}%)
- **Características**: {len(self.selected_features)}
- **Clases**: {', '.join(self.target_encoder.classes_)}

## Archivos Generados
- **{filename}**: Modelo principal (usar en Streamlit)
- **{debug_filename}**: Información técnica detallada

## Características del Modelo
{chr(10).join([f"- {feature}" for feature in self.selected_features])}

## Instrucciones para Google Drive
1. Sube `{filename}` a Google Drive
2. Comparte el archivo y obtén el file ID
3. Usa el file ID en tu aplicación Streamlit

## Clases de Salida
{chr(10).join([f"- {cls}" for cls in self.target_encoder.classes_])}
"""
        
        readme_filename = f'README_{timestamp}.md'
        with open(readme_filename, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        print(f"✅ README guardado: {readme_filename}")
        
        return filename, debug_filename, readme_filename
    
    def validar_modelo(self, filename):
        """Valida que el modelo se pueda cargar correctamente"""
        print(f"🔍 VALIDANDO MODELO: {filename}")
        print("=" * 50)
        
        try:
            # Cargar modelo
            modelo_components = joblib.load(filename)
            
            # Verificar componentes
            required = ['modelo', 'scaler', 'encoders', 'selected_features', 'target_encoder']
            missing = [comp for comp in required if comp not in modelo_components]
            
            if missing:
                print(f"❌ Componentes faltantes: {missing}")
                return False
            
            # Test de predicción
            modelo = modelo_components['modelo']
            scaler = modelo_components['scaler']
            n_features = len(modelo_components['selected_features'])
            
            test_data = np.random.randn(1, n_features)
            test_scaled = scaler.transform(test_data)
            prediction = modelo.predict(test_scaled)
            
            print(f"✅ Validación exitosa")
            print(f"✅ Características: {n_features}")
            print(f"✅ Clases: {modelo_components['target_encoder'].classes_}")
            print(f"✅ Accuracy: {modelo_components.get('accuracy', 'N/A')}")
            print(f"✅ Test prediction: {prediction[0]}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error en validación: {str(e)}")
            return False
    
    def entrenar_completo(self, n_features=15, sample_size=0.8):
        """Ejecuta el proceso completo de entrenamiento"""
        print("🎯 INICIANDO ENTRENAMIENTO COMPLETO")
        print("=" * 60)
        
        try:
            # 1. Configuración y carga de datos
            self.setup_kaggle()
            self.load_data()
            
            # 2. Análisis exploratorio
            self.analisis_exploratorio()
            
            # 3. Preparación de datos
            X_train, X_test, y_train, y_test = self.preparar_datos(n_features)
            
            # 4. Entrenamiento
            self.entrenar_modelo(X_train, y_train, sample_size)
            
            # 5. Evaluación
            self.evaluar_modelo(X_test, y_test)
            
            # 6. Guardado
            filename, debug_file, readme_file = self.guardar_modelo_completo()
            
            # 7. Validación
            if self.validar_modelo(filename):
                print(f"\n🎉 PROCESO COMPLETADO EXITOSAMENTE!")
                print(f"📊 Accuracy final: {self.accuracy:.2%}")
                print(f"💾 Archivo principal: {filename}")
                print(f"📋 Documentación: {readme_file}")
                print(f"\n📌 PRÓXIMOS PASOS:")
                print(f"1. Descarga el archivo {filename}")
                print(f"2. Súbelo a Google Drive")
                print(f"3. Comparte el archivo y obtén el file ID")
                print(f"4. Usa el file ID en tu aplicación Streamlit")
                
                return filename
            else:
                print(f"❌ Error en la validación")
                return None
                
        except Exception as e:
            print(f"❌ Error en el proceso: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

# Función principal para ejecutar
def main():
    """Función principal"""
    trainer = PersonalityModelTrainer()
    
    # Configuración del entrenamiento
    N_FEATURES = 15  # Número de características a seleccionar
    SAMPLE_SIZE = 0.8  # Porcentaje del dataset a usar (para datasets grandes)
    
    # Ejecutar entrenamiento completo
    modelo_file = trainer.entrenar_completo(
        n_features=N_FEATURES,
        sample_size=SAMPLE_SIZE
    )
    
    if modelo_file:
        print(f"\n✨ Modelo '{modelo_file}' listo para Google Drive!")
        
        # Descargar archivo automáticamente
        print("\n📥 Descargando archivo...")
        files.download(modelo_file)
    else:
        print("\n💥 Error en el entrenamiento. Revisa los logs.")

# Ejecutar el programa
if __name__ == "__main__":
    main()
