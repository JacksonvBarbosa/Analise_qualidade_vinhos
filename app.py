"""
Streamlit App para Predição de Qualidade de Vinhos
App de produção para uso pelos funcionários da empresa
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Adiciona o diretório src ao path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from analise_qualidade_vinhos.pipeline.predict import load_model, predict_from_dataframe
from analise_qualidade_vinhos.config.settings import MODEL_DIR

# Configuração da página
st.set_page_config(
    page_title="🍷 Predição de Qualidade de Vinhos",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS personalizado
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #8B0000;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        margin: 2rem 0;
        text-align: center;
    }
    .high-quality {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    .low-quality {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    .stButton>button {
        width: 100%;
        background-color: #8B0000;
        color: white;
        font-size: 1.2rem;
        padding: 0.75rem;
        border-radius: 8px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #A52A2A;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🍷 Sistema de Predição de Qualidade de Vinhos</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Insira as características físico-químicas do vinho para prever sua qualidade</p>', unsafe_allow_html=True)

# Sidebar com informações
with st.sidebar:
    st.image("https://via.placeholder.com/200x100/8B0000/FFFFFF?text=JACKWine", use_container_width=True)
    st.markdown("### 📊 Sobre o Sistema")
    st.info("""
    Este sistema utiliza Machine Learning para prever a qualidade do vinho baseado em 
    características físico-químicas. O modelo foi treinado com dados históricos e 
    classifica os vinhos em:
    - **Alta qualidade** (≥ 6)
    - **Baixa qualidade** (< 6)
    """)
    
    st.markdown("### 📝 Instruções")
    st.markdown("""
    1. Preencha todos os campos no formulário
    2. Clique em "Prever Qualidade"
    3. Visualize o resultado e recomendações
    """)
    
    st.markdown("### ⚠️ Valores de Referência")
    st.markdown("""
    - **Álcool**: 8-15% (ideal: 10-13%)
    - **pH**: 2.8-4.0 (ideal: 3.0-3.5)
    - **Sulfatos**: 0.3-2.0 g/L (ideal: 0.5-1.0)
    - **Acidez Volátil**: < 1.0 g/L
    """)

# Carregar modelo (com cache)
@st.cache_resource
def load_wine_model():
    """Carrega o modelo treinado."""
    model_path = MODEL_DIR / "wine_quality_model.joblib"
    if not model_path.exists():
        st.error("⚠️ Modelo não encontrado! Por favor, treine o modelo primeiro.")
        st.stop()
    return load_model(model_path)

try:
    model = load_wine_model()
except Exception as e:
    st.error(f"Erro ao carregar modelo: {e}")
    st.stop()

# Formulário principal
st.markdown("### 📋 Características do Vinho")

# Organizar em colunas para melhor UX
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### 🧪 Propriedades Químicas")
    fixed_acidity = st.number_input(
        "Acidez Fixa (g/L)",
        min_value=0.0,
        max_value=20.0,
        value=7.0,
        step=0.1,
        help="Ácido tartárico, geralmente entre 4-10 g/L"
    )
    
    volatile_acidity = st.number_input(
        "Acidez Volátil (g/L)",
        min_value=0.0,
        max_value=2.0,
        value=0.5,
        step=0.01,
        help="Ácido acético, idealmente < 1.0 g/L"
    )
    
    citric_acid = st.number_input(
        "Ácido Cítrico (g/L)",
        min_value=0.0,
        max_value=2.0,
        value=0.3,
        step=0.01,
        help="Ajuda na frescura, idealmente 0.2-0.5 g/L"
    )
    
    residual_sugar = st.number_input(
        "Açúcar Residual (g/L)",
        min_value=0.0,
        max_value=20.0,
        value=2.0,
        step=0.1,
        help="Açúcar restante após fermentação"
    )
    
    chlorides = st.number_input(
        "Cloretos (g/L)",
        min_value=0.0,
        max_value=1.0,
        value=0.08,
        step=0.01,
        help="Salinidade, idealmente 0.05-0.15 g/L"
    )

with col2:
    st.markdown("#### 🧬 Enxofre e pH")
    free_sulfur_dioxide = st.number_input(
        "Dióxido de Enxofre Livre (mg/L)",
        min_value=0.0,
        max_value=100.0,
        value=15.0,
        step=1.0,
        help="SO₂ livre, preservante, idealmente 10-30 mg/L"
    )
    
    total_sulfur_dioxide = st.number_input(
        "Dióxido de Enxofre Total (mg/L)",
        min_value=0.0,
        max_value=300.0,
        value=45.0,
        step=1.0,
        help="SO₂ total, idealmente 30-100 mg/L"
    )
    
    density = st.number_input(
        "Densidade (g/cm³)",
        min_value=0.990,
        max_value=1.010,
        value=0.997,
        step=0.001,
        format="%.3f",
        help="Densidade do vinho, geralmente 0.990-1.000"
    )
    
    ph = st.number_input(
        "pH",
        min_value=2.5,
        max_value=4.5,
        value=3.3,
        step=0.01,
        help="Acidez, idealmente 3.0-3.5"
    )
    
    sulphates = st.number_input(
        "Sulfatos (g/L)",
        min_value=0.0,
        max_value=2.0,
        value=0.65,
        step=0.01,
        help="Aditivo, idealmente 0.5-1.0 g/L"
    )

with col3:
    st.markdown("#### 🍇 Propriedades Físicas")
    alcohol = st.number_input(
        "Teor Alcoólico (%)",
        min_value=8.0,
        max_value=16.0,
        value=10.5,
        step=0.1,
        help="Teor alcoólico, idealmente 10-13%"
    )
    
    st.markdown("---")
    st.markdown("### 📊 Visualização Rápida")
    
    # Gráfico de barras simples para visualização
    chart_data = pd.DataFrame({
        'Métrica': ['Álcool', 'pH', 'Sulfatos', 'Acidez Volátil'],
        'Valor': [alcohol, ph, sulphates, volatile_acidity],
        'Ideal Min': [10, 3.0, 0.5, 0],
        'Ideal Max': [13, 3.5, 1.0, 1.0]
    })
    
    st.bar_chart(chart_data.set_index('Métrica')[['Valor', 'Ideal Min', 'Ideal Max']])

# Botão de predição
st.markdown("---")
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])

with col_btn2:
    predict_button = st.button("🔮 Prever Qualidade do Vinho", type="primary", use_container_width=True)

# Processar predição
if predict_button:
    with st.spinner("🔄 Processando predição..."):
        try:
            # Criar DataFrame com os dados inseridos
            wine_data = pd.DataFrame({
                "fixed_acidity": [fixed_acidity],
                "volatile_acidity": [volatile_acidity],
                "citric_acid": [citric_acid],
                "residual_sugar": [residual_sugar],
                "chlorides": [chlorides],
                "free_sulfur_dioxide": [free_sulfur_dioxide],
                "total_sulfur_dioxide": [total_sulfur_dioxide],
                "density": [density],
                "ph": [ph],
                "sulphates": [sulphates],
                "alcohol": [alcohol],
            })
            
            # Fazer predição
            predictions = predict_from_dataframe(model, wine_data)
            prediction = predictions[0]
            
            # Exibir resultado
            st.markdown("---")
            
            if prediction == "Alta qualidade":
                st.markdown(
                    f'<div class="prediction-box high-quality">'
                    f'<h2>✅ Alta Qualidade</h2>'
                    f'<p style="font-size: 1.5rem;">Este vinho apresenta características de <strong>alta qualidade</strong>!</p>'
                    f'</div>',
                    unsafe_allow_html=True
                )
                
                st.success("🎉 **Recomendação**: Este vinho está aprovado para produção e comercialização!")
                
                # Métricas de qualidade
                st.markdown("### 📈 Análise das Características")
                col_met1, col_met2, col_met3, col_met4 = st.columns(4)
                
                with col_met1:
                    st.metric("Álcool", f"{alcohol}%", "✅ Ideal" if 10 <= alcohol <= 13 else "⚠️ Fora do ideal")
                with col_met2:
                    st.metric("pH", f"{ph:.2f}", "✅ Ideal" if 3.0 <= ph <= 3.5 else "⚠️ Fora do ideal")
                with col_met3:
                    st.metric("Sulfatos", f"{sulphates} g/L", "✅ Ideal" if 0.5 <= sulphates <= 1.0 else "⚠️ Fora do ideal")
                with col_met4:
                    st.metric("Acidez Volátil", f"{volatile_acidity} g/L", "✅ Ideal" if volatile_acidity < 1.0 else "⚠️ Alta")
                
            else:
                st.markdown(
                    f'<div class="prediction-box low-quality">'
                    f'<h2>⚠️ Baixa Qualidade</h2>'
                    f'<p style="font-size: 1.5rem;">Este vinho apresenta características de <strong>baixa qualidade</strong>.</p>'
                    f'</div>',
                    unsafe_allow_html=True
                )
                
                st.warning("⚠️ **Recomendação**: Este vinho precisa de ajustes antes da produção.")
                
                # Sugestões de melhoria
                st.markdown("### 💡 Sugestões de Melhoria")
                suggestions = []
                
                if alcohol < 10:
                    suggestions.append("🔺 **Aumentar teor alcoólico** para 10-13%")
                elif alcohol > 13:
                    suggestions.append("🔻 **Reduzir teor alcoólico** para 10-13%")
                
                if volatile_acidity >= 1.0:
                    suggestions.append("🔻 **Reduzir acidez volátil** para < 1.0 g/L")
                
                if ph < 3.0 or ph > 3.5:
                    suggestions.append("⚖️ **Ajustar pH** para 3.0-3.5")
                
                if sulphates < 0.5:
                    suggestions.append("🔺 **Aumentar sulfatos** para 0.5-1.0 g/L")
                elif sulphates > 1.0:
                    suggestions.append("🔻 **Reduzir sulfatos** para 0.5-1.0 g/L")
                
                if suggestions:
                    for suggestion in suggestions:
                        st.markdown(f"- {suggestion}")
                else:
                    st.info("Verifique outros parâmetros ou consulte um especialista.")
            
            # Informações adicionais
            with st.expander("📊 Ver Dados Inseridos"):
                st.dataframe(wine_data.T, use_container_width=True)
                
        except Exception as e:
            st.error(f"❌ Erro ao processar predição: {e}")
            st.exception(e)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <p>🍷 Sistema de Predição de Qualidade de Vinhos - JACKWine</p>
        <p>Desenvolvido com Machine Learning | Versão 1.0</p>
    </div>
    """,
    unsafe_allow_html=True
)

