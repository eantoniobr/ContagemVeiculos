@echo off
chcp 65001 >nul
title Sistema Inteligente de Contagem de Veículos por Rota
cd /d "%~dp0"
echo ============================================================
echo   Iniciando Sistema de Rastreamento e Contagem de Veiculos
echo ============================================================
python main.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Ocorreu um erro ao executar a aplicacao.
    pause
)
