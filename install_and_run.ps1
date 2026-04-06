# ══════════════════════════════════════════════════════════
# HX-AM v4 — установка зависимостей и запуск
# Запускать из папки проекта: cd путь\к\проекту
# ══════════════════════════════════════════════════════════

# ШАГ 1 — Проверка Python
Write-Host "`n[1/4] Проверка Python..." -ForegroundColor Cyan
python --version
if ($LASTEXITCODE -ne 0) {
    Write-Host "Python не найден. Установи Python 3.10+ с python.org" -ForegroundColor Red
    exit 1
}

# ШАГ 2 — Создание виртуального окружения (если не существует)
Write-Host "`n[2/4] Виртуальное окружение..." -ForegroundColor Cyan
if (-Not (Test-Path "venv")) {
    python -m venv venv
    Write-Host "Создано: venv\" -ForegroundColor Green
} else {
    Write-Host "Уже существует: venv\" -ForegroundColor Yellow
}

# ШАГ 3 — Активация и установка зависимостей
Write-Host "`n[3/4] Установка зависимостей..." -ForegroundColor Cyan
.\venv\Scripts\Activate.ps1
pip install --upgrade pip --quiet
pip install -r requirements.txt

if ($LASTEXITCODE -ne 0) {
    Write-Host "Ошибка установки. Проверь requirements.txt" -ForegroundColor Red
    exit 1
}
Write-Host "Все зависимости установлены." -ForegroundColor Green

# ШАГ 4 — Создание папок если их нет
Write-Host "`n[4/4] Подготовка папок..." -ForegroundColor Cyan
$folders = @("artifacts", "chat_history", "prompts")
foreach ($folder in $folders) {
    if (-Not (Test-Path $folder)) {
        New-Item -ItemType Directory -Path $folder | Out-Null
        Write-Host "Создана папка: $folder\" -ForegroundColor Green
    } else {
        Write-Host "Уже существует: $folder\" -ForegroundColor Yellow
    }
}

Write-Host "`n✅ Готово! Запускаю сервер..." -ForegroundColor Green
Write-Host "   Открой браузер: http://localhost:8000" -ForegroundColor White
Write-Host "   Для остановки: Ctrl+C`n" -ForegroundColor White

python hxam_v_4_server.py
