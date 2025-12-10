@echo off
echo ====================================
echo  App Streamlit - Qualidade Vinhos
echo ====================================
echo.
echo Ativando ambiente virtual...
call .venv\Scripts\activate.bat
echo.
echo Iniciando app Streamlit...
streamlit run app.py
pause



