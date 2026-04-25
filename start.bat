@echo off
REM ============================================
REM GraphRAG Local LLM - 一括起動スクリプト
REM ============================================
REM 前提:
REM   - Anaconda がインストール済み
REM   - conda環境 "graphrag" が作成済み
REM   - Ollama がインストール済み・起動済み
REM   - Apache (XAMPP等) がインストール済み
REM   - Node.js がインストール済み
REM ============================================

echo [1/4] Ollama起動確認...
ollama list >nul 2>&1
if errorlevel 1 (
    echo   Ollamaが見つかりません。先に起動してください。
    pause
    exit /b 1
)
echo   OK

echo [2/4] FastAPIバックエンド起動...
start "FastAPI" cmd /k "cd /d %~dp0backend && conda activate graphrag && uvicorn main:app --host 0.0.0.0 --port 8000 --reload"
timeout /t 3 /nobreak >nul

echo [3/4] Reactフロントエンド起動...
start "React" cmd /k "cd /d %~dp0frontend && npm start"
timeout /t 5 /nobreak >nul

echo [4/4] Apache再起動...
REM XAMPP の場合:
REM   "C:\xampp\apache\bin\httpd.exe" -k restart
REM Apache単体の場合:
REM   httpd -k restart
echo   Apache設定は手動で確認してください

echo.
echo ============================================
echo 起動完了！
echo   React:   http://localhost:3000
echo   Apache:  http://localhost
echo   API:     http://localhost:8000/docs
echo ============================================
pause
