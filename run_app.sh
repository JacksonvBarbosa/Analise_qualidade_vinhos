#!/bin/bash
echo "===================================="
echo " App Streamlit - Qualidade Vinhos"
echo "===================================="
echo ""
echo "Ativando ambiente virtual..."
source .venv/bin/activate
echo ""
echo "Iniciando app Streamlit..."
streamlit run app.py

