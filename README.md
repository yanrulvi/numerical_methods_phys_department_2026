# numerical_methods_phys_department_2026
This repository stores problems and their solutions in numerical methods, Physics Faculty course, 2026

## Установка и запуск
### Требования
Python 3.8 или выше

## Настройка окружения
### 1. Клонирование репозитория
```bash
git clone https://github.com/yanrulvi/numerical_methods_phys_department_2026
cd numerical_methods_phys_department_2026
```
### 2. Создание виртуального окружения
```bash
python -m venv venv
```
### 3. Активация окружения
#### Windows (CMD):
```bash
venv\Scripts\activate
```
#### Windows (PowerShell):
```bash
venv\Scripts\Activate.ps1
```
#### macOS / Linux:
```bash
source venv/bin/activate
```
### 4. Установка зависимостей
```bash
pip install -r requirements.txt
```
### 5. Деактивация (после завершения работы)
```bash
deactivate
```
## Быстрый старт (все команды сразу)
### Windows:

```bash
python -m venv venv && venv\Scripts\activate && pip install -r requirements.txt
```
### macOS / Linux:

```bash
python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt
```
## Обновление зависимостей
После установки новых пакетов обновите requirements.txt:

```bash
pip freeze > requirements.txt
```
