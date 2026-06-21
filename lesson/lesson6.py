"""
================================================================================
УРОК: Линейная регрессия с несколькими переменными + Train/Test Split
АВТОР: Преподаватель курса "Математика для Data Science"
================================================================================

Этот файл содержит полный код для практического занятия.
Каждая строчка прокомментирована для понимания.
"""

# ==============================================================================
# БЛОК 1: ИМПОРТ БИБЛИОТЕК
# ==============================================================================

# pandas - для работы с таблицами (чтение CSV, фильтрация, группировка)
import pandas as pd

# numpy - для математических операций (массивы, корни, логарифмы)
import numpy as np

# matplotlib и seaborn - для визуализации (графики, диаграммы)
import matplotlib.pyplot as plt
import seaborn as sns

# sklearn.model_selection - для разбиения данных на train/test
from sklearn.model_selection import train_test_split

# sklearn.preprocessing - для масштабирования и кодирования категорий
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# sklearn.compose - для объединения разных преобразований в один блок
from sklearn.compose import ColumnTransformer

# sklearn.pipeline - для создания конвейера обработки и обучения
from sklearn.pipeline import Pipeline

# sklearn.linear_model - сама модель линейной регрессии
from sklearn.linear_model import LinearRegression

# sklearn.metrics - метрики качества (MSE, MAE, R²)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Отключаем предупреждения (чтобы не отвлекали)
import warnings

warnings.filterwarnings('ignore')

print("✅ Все библиотеки загружены!")
print("=" * 60)

# ==============================================================================
# БЛОК 2: СОЗДАНИЕ СИНТЕТИЧЕСКИХ ДАННЫХ (если нет реального датасета)
# ==============================================================================

"""
ПОЧЕМУ СИНТЕТИЧЕСКИЕ ДАННЫЕ?
- Мы точно знаем правильную зависимость (y = 5000*x1 - 2000*x2 + 300*x3 - 500000)
- Можем проверить, насколько хорошо модель восстановила эти коэффициенты
- Легко объяснить студентам, потому что мы создали правила сами
"""

# Фиксируем генератор случайных чисел (чтобы результат был воспроизводимым)
np.random.seed(42)

# Количество объектов в наборе данных
n_samples = 500

# Создаем DataFrame с признаками
# Каждый признак - это колонка с разным распределением
data = pd.DataFrame({
    # ПРИЗНАК 1: Площадь дома (от 30 до 150 кв.м, равномерное распределение)
    'area': np.random.uniform(30, 150, n_samples),

    # ПРИЗНАК 2: Этаж (от 1 до 10, целые числа)
    'floor': np.random.randint(1, 11, n_samples),

    # ПРИЗНАК 3: Год постройки (от 1980 до 2023)
    'year': np.random.randint(1980, 2024, n_samples),

    # ПРИЗНАК 4: Тип дома (категориальный признак)
    'house_type': np.random.choice(['panel', 'brick', 'monolith'], n_samples),

    # ПРИЗНАК 5: Район (категориальный)
    'district': np.random.choice(['center', 'north', 'south', 'east', 'west'], n_samples)
})

print("📊 Созданы признаки:")
print(data.head())
print("\n📈 Статистика числовых признаков:")
print(data.describe())
print("=" * 60)

# ==============================================================================
# БЛОК 3: ФОРМИРОВАНИЕ ЦЕЛЕВОЙ ПЕРЕМЕННОЙ (ЦЕНА)
# ==============================================================================

"""
СОЗДАЁМ ПРАВИЛЬНУЮ ЗАВИСИМОСТЬ С ШУМОМ:
Реальная цена = (влияние площади) + (влияние этажа) + (влияние года) + шум

ПОЧЕМУ ДОБАВЛЯЕМ ШУМ?
- В реальной жизни цена зависит не только от этих факторов
- Шум делает задачу реалистичной (нет идеальной линейной зависимости)
"""

# Истинные коэффициенты (о которых модель не знает)
# Мы их используем потом для проверки, насколько хорошо обучилась модель
TRUE_COEF_AREA = 5000  # Каждый кв.м добавляет 5000 руб
TRUE_COEF_FLOOR = -2000  # Каждый этаж выше первого уменьшает цену на 2000 руб
TRUE_COEF_YEAR = 300  # Каждый новый год прибавляет 300 руб
TRUE_INTERCEPT = -500000  # Базовая константа

# Рассчитываем "идеальную" цену без шума (только по формуле)
ideal_price = (TRUE_COEF_AREA * data['area'] +
               TRUE_COEF_FLOOR * data['floor'] +
               TRUE_COEF_YEAR * data['year'] +
               TRUE_INTERCEPT)

# Добавляем шум (нормальное распределение со средним 0 и стандартным отклонением 50000)
# Стандартное отклонение = 50000 означает, что ошибка модели может быть до ±150000
noise = np.random.normal(0, 50000, n_samples)

# Итоговая цена = идеальная формула + случайный шум
data['price'] = ideal_price + noise

# Преобразуем цену в целые числа (для реалистичности)
data['price'] = data['price'].astype(int)

print("\n💰 Целевая переменная создана!")
print(f"Диапазон цен: от {data['price'].min():,} до {data['price'].max():,} руб")
print(f"Средняя цена: {data['price'].mean():,.0f} руб")
print("=" * 60)

# ==============================================================================
# БЛОК 4: ИССЛЕДОВАТЕЛЬСКИЙ АНАЛИЗ ДАННЫХ (EDA)
# ==============================================================================

"""
EDA - ЭТО ПЕРВЫЙ ШАГ ЛЮБОГО ПРОЕКТА!
Зачем? Чтобы:
1. Понять структуру данных
2. Найти выбросы и аномалии
3. Посмотреть распределение целевой переменной
4. Оценить корреляции между признаками
"""

print("\n📊 ИССЛЕДОВАТЕЛЬСКИЙ АНАЛИЗ ДАННЫХ")
print("-" * 40)

# 1. Информация о типах данных и пропусках
print("\n1. Информация о DataFrame:")
print(data.info())

# 2. Проверка на пропуски (None, NaN)
print("\n2. Количество пропусков в каждом столбце:")
print(data.isnull().sum())

# 3. Статистика числовых признаков
print("\n3. Статистика числовых признаков:")
print(data.describe())

# 4. Статистика категориальных признаков
print("\n4. Статистика категориальных признаков:")
categorical_cols = ['house_type', 'district']
for col in categorical_cols:
    print(f"\n{col}:")
    print(data[col].value_counts())

# 5. ВИЗУАЛИЗАЦИЯ РАСПРЕДЕЛЕНИЙ
plt.figure(figsize=(15, 10))

# 5.1. Гистограмма цен (насколько скошено распределение)
plt.subplot(2, 3, 1)
plt.hist(data['price'], bins=50, edgecolor='black', alpha=0.7)
plt.xlabel('Цена (руб)')
plt.ylabel('Количество')
plt.title('Распределение цен на недвижимость')
plt.axvline(data['price'].mean(), color='red', linestyle='--', label='Среднее')
plt.axvline(data['price'].median(), color='green', linestyle='--', label='Медиана')
plt.legend()

# 5.2. Boxplot цены (для поиска выбросов)
plt.subplot(2, 3, 2)
plt.boxplot(data['price'], vert=True)
plt.ylabel('Цена (руб)')
plt.title('Boxplot цен (выбросы выше/ниже усов)')

# 5.3. Зависимость цены от площади
plt.subplot(2, 3, 3)
plt.scatter(data['area'], data['price'], alpha=0.5)
plt.xlabel('Площадь (кв.м)')
plt.ylabel('Цена (руб)')
plt.title('Цена vs Площадь')
# Добавляем линию тренда
z = np.polyfit(data['area'], data['price'], 1)
p = np.poly1d(z)
plt.plot(data['area'].sort_values(), p(data['area'].sort_values()), "r--", alpha=0.8)

# 5.4. Зависимость цены от года постройки
plt.subplot(2, 3, 4)
plt.scatter(data['year'], data['price'], alpha=0.5)
plt.xlabel('Год постройки')
plt.ylabel('Цена (руб)')
plt.title('Цена vs Год постройки')

# 5.5. Цена по типам домов (boxplot)
plt.subplot(2, 3, 5)
data.boxplot(column='price', by='house_type', ax=plt.gca())
plt.title('Цена по типу дома')
plt.suptitle('')  # Убираем автоматический заголовок

# 5.6. Цена по районам (boxplot)
plt.subplot(2, 3, 6)
data.boxplot(column='price', by='district', ax=plt.gca())
plt.xticks(rotation=45)
plt.title('Цена по районам')
plt.suptitle('')

plt.tight_layout()
plt.show()

# 6. КОРРЕЛЯЦИОННАЯ МАТРИЦА (только для числовых признаков)
print("\n5. Корреляция числовых признаков с ценой:")
numeric_features = ['area', 'floor', 'year', 'price']
correlation_matrix = data[numeric_features].corr()
print(correlation_matrix['price'].sort_values(ascending=False))

# Визуализация корреляционной матрицы
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Корреляционная матрица')
plt.show()

print("=" * 60)

# ==============================================================================
# БЛОК 5: РАЗДЕЛЕНИЕ НА ОБУЧАЮЩУЮ И ТЕСТОВУЮ ВЫБОРКИ
# ==============================================================================

"""
ГЛАВНОЕ ПРАВИЛО МАШИННОГО ОБУЧЕНИЯ:
НИКОГДА не используйте одни и те же данные для обучения и тестирования!

ПОЧЕМУ?
- Модель запомнит ответы и покажет отличное качество на обучении
- Но на новых данных (которые не видела) провалится
- Это называется "переобучение" (overfitting)

КАК ПРАВИЛЬНО?
1. Отделяем целевую переменную (y) от признаков (X)
2. Разбиваем на train (70-80%) и test (20-30%)
3. Обучаем ТОЛЬКО на train
4. Проверяем ТОЛЬКО на test
"""

# Шаг 1: Отделяем признаки (X) от целевой переменной (y)
# X - все колонки, КРОМЕ 'price'
X = data.drop('price', axis=1)

# y - только колонка 'price' (то, что предсказываем)
y = data['price']

print("\n🎯 РАЗДЕЛЕНИЕ НА TRAIN И TEST")
print("-" * 40)
print(f"Размер полного набора: {len(data)} объектов")
print(f"Количество признаков (X): {X.shape[1]}")
print(f"Целевая переменная (y): предсказание цены")

# Шаг 2: Разбиваем с параметрами:
# - test_size=0.2: 20% данных пойдёт на тест, 80% на обучение
# - random_state=42: фиксируем случайность (чтобы результат был одинаковым при каждом запуске)
# - stratify: не используем (только для классификации, а у нас регрессия)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,  # 20% на тест (обычно 20-30%)
    random_state=42  # Для воспроизводимости
)

print(f"\n📊 Результат разбиения:")
print(f"Обучающая выборка (train): {len(X_train)} объектов ({len(X_train) / len(data) * 100:.0f}%)")
print(f"Тестовая выборка (test): {len(X_test)} объектов ({len(X_test) / len(data) * 100:.0f}%)")

# Проверяем, что разбиение произошло случайно (распределение цен похоже)
print(f"\nСтатистика цены в обучающей выборке:")
print(f"  Среднее: {y_train.mean():,.0f} руб")
print(f"  Медиана: {y_train.median():,.0f} руб")
print(f"\nСтатистика цены в тестовой выборке:")
print(f"  Среднее: {y_test.mean():,.0f} руб")
print(f"  Медиана: {y_test.median():,.0f} руб")

print("=" * 60)

# ==============================================================================
# БЛОК 6: ПОДГОТОВКА ПРЕПРОЦЕССОРА (ColumnTransformer)
# ==============================================================================

"""
ЧТО ТАКОЕ ПРЕПРОЦЕССОР?
Это "конвейер" обработки данных, который:
- Для числовых признаков: масштабирует их (StandardScaler)
- Для категориальных: превращает в 0/1 векторы (OneHotEncoder)

ПОЧЕМУ НЕЛЬЗЯ ПРОСТО ПОДАТЬ ДАННЫЕ В МОДЕЛЬ?
1. Разные масштабы: площадь (30-150) и год (1980-2023)
   - Большие числа "давят" на маленькие
   - Модель решит, что год важнее, хотя это не так
2. Категории: "panel", "brick" - это текст, модель не понимает
"""

# Определяем, какие колонки у нас числовые, а какие категориальные
numeric_features = ['area', 'floor', 'year']  # Числовые признаки
categorical_features = ['house_type', 'district']  # Категориальные признаки

print("\n🔄 СОЗДАНИЕ ПРЕПРОЦЕССОРА")
print("-" * 40)
print(f"Числовые признаки: {numeric_features}")
print(f"Категориальные признаки: {categorical_features}")

# Создаём препроцессор, который применяет разные обработки к разным колонкам
preprocessor = ColumnTransformer([
    # Для числовых признаков: StandardScaler (нормализация к среднему=0, дисперсии=1)
    # Что делает StandardScaler: (x - mean) / std
    # Результат: все признаки в одном масштабе
    ('numeric', StandardScaler(), numeric_features),

    # Для категориальных признаков: OneHotEncoder
    # Что делает OneHotEncoder: создаёт отдельную колонку для каждой категории
    # Например, house_type = ['panel', 'brick', 'monolith'] превращается в 3 колонки
    #   panel: [1, 0, 0]
    #   brick: [0, 1, 0]
    #   monolith: [0, 0, 1]
    # handle_unknown='ignore' - если встретится новая категория в тесте, просто игнорируем
    ('categorical', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
])

# Важно: preprocessor только ОПИСЫВАЕТ преобразования, но не применяет их
print("\n✅ Препроцессор создан, но не применён (будет вызван в Pipeline)")

# Демонстрация того, как работает OneHotEncoder (на примере)
sample_data = X_train[['house_type']].head()
encoder_demo = OneHotEncoder(sparse_output=False)
encoded_demo = encoder_demo.fit_transform(sample_data)
print(f"\n📚 Пример работы OneHotEncoder (тип дома):")
print(f"Исходные данные:\n{sample_data}")
print(f"\nПосле OneHotEncoder:\n{encoded_demo}")
print(f"Категории: {encoder_demo.categories_[0]}")

print("=" * 60)

# ==============================================================================
# БЛОК 7: СОЗДАНИЕ FULL PIPELINE (ВСЕ ОБРАБОТКИ + МОДЕЛЬ)
# ==============================================================================

"""
ЧТО ТАКОЕ PIPELINE?
Это конвейер, который:
1. Принимает сырые данные
2. Применяет предобработку (масштабирование, OneHotEncoder)
3. Передаёт обработанные данные в модель
4. Возвращает предсказание

ПРЕИМУЩЕСТВА PIPELINE:
1. Код становится чистым (всё в одном месте)
2. Нельзя случайно "утечь" данные (fit_transform на train, transform на test)
3. Легко переключать модели (меняем последний шаг)
4. Удобно для кросс-валидации
"""

print("\n🚀 СОЗДАНИЕ FULL PIPELINE")
print("-" * 40)

# Создаём пайплайн из двух шагов:
pipeline = Pipeline([
    # Шаг 1: Предобработка данных (масштабирование + OneHotEncoder)
    ('preprocessor', preprocessor),

    # Шаг 2: Модель линейной регрессии
    ('regressor', LinearRegression())
])

print("Pipeline создан со следующей структурой:")
print("   Шаг 1: preprocessor (ColumnTransformer)")
print("      - numeric → StandardScaler")
print("      - categorical → OneHotEncoder")
print("   Шаг 2: regressor → LinearRegression")

# Показываем, что внутри пайплайна (для любопытных)
print(f"\n📋 Детали pipeline:")
print(pipeline)

print("=" * 60)

# ==============================================================================
# БЛОК 8: ОБУЧЕНИЕ МОДЕЛИ
# ==============================================================================

"""
ПРОЦЕСС ОБУЧЕНИЯ:
1. Pipeline автоматически:
   - Применяет fit_transform к предобработчику на обучающих данных
   - Обучает модель на обработанных данных
2. Мы НЕ вызываем отдельно fit для scaler и модели
3. Всё происходит в одной строке!
"""

print("\n🏋️ ОБУЧЕНИЕ МОДЕЛИ")
print("-" * 40)

# Обучаем пайплайн на обучающей выборке
# .fit() - основная команда обучения
pipeline.fit(X_train, y_train)

print("✅ Модель обучена успешно!")

# После обучения мы можем посмотреть:
# 1. Коэффициенты модели (веса признаков)
# 2. Смещение (intercept)

# Достаём обученную модель регрессии из пайплайна
trained_model = pipeline.named_steps['regressor']

# Достаём обученный препроцессор
trained_preprocessor = pipeline.named_steps['preprocessor']

print(f"\n📊 Коэффициенты модели (веса):")
# .coef_ - массив весов для каждого признака ПОСЛЕ предобработки
print(f"Веса (w): {trained_model.coef_}")

print(f"\n📊 Смещение модели (intercept):")
print(f"Смещение (b): {trained_model.intercept_:.2f}")

# ВАЖНО: количество весов = (числовые признаки) + (все категории из OneHot)
n_numeric = len(numeric_features)
n_categories = (len(trained_preprocessor.named_transformers_['categorical'].categories_[0]) +
                len(trained_preprocessor.named_transformers_['categorical'].categories_[1]))
print(f"\n📌 Пояснение:")
print(f"Всего весов: {len(trained_model.coef_)}")
print(f"  - {n_numeric} весов от числовых признаков (area, floor, year)")
print(f"  - {n_categories} весов от категориальных признаков (one-hot кодирование)")

print("=" * 60)

# ==============================================================================
# БЛОК 9: ПРЕДСКАЗАНИЕ И ОЦЕНКА КАЧЕСТВА
# ==============================================================================

"""
МЕТРИКИ КАЧЕСТВА ДЛЯ РЕГРЕССИИ:

1. MAE (Mean Absolute Error) - средняя абсолютная ошибка
   - Формула: (1/n) * Σ|y_true - y_pred|
   - Плюс: интерпретируема (ошибка в тех же единицах, что и цена)
   - Минус: не штрафует большие ошибки сильнее

2. MSE (Mean Squared Error) - средняя квадратичная ошибка
   - Формула: (1/n) * Σ(y_true - y_pred)²
   - Плюс: штрафует большие ошибки (квадрат)
   - Минус: единицы измерения в квадрате

3. RMSE (Root Mean Squared Error) - корень из MSE
   - Формула: √MSE
   - Плюс: единицы измерения как у целевой переменной

4. R² (Коэффициент детерминации) - доля объяснённой дисперсии
   - Формула: 1 - (MSE_модели / MSE_базовой_модели)
   - Диапазон: (-∞, 1]
   - R² = 1: идеальное предсказание
   - R² = 0: модель не лучше, чем предсказание среднего
   - R² < 0: модель хуже, чем предсказание среднего
"""

print("\n📈 ПРЕДСКАЗАНИЕ И ОЦЕНКА КАЧЕСТВА")
print("-" * 40)

# Делаем предсказания на обучающей выборке
# На обучающей мы смотрим, не переобучилась ли модель
y_train_pred = pipeline.predict(X_train)

# Делаем предсказания на тестовой выборке
# На тестовой мы оцениваем РЕАЛЬНУЮ способность модели к обобщению
y_test_pred = pipeline.predict(X_test)


# Функция для расчёта всех метрик (чтобы не повторять код)
def calculate_metrics(y_true, y_pred, dataset_name):
    """Рассчитывает MAE, MSE, RMSE, R² для предсказаний"""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    print(f"\n{'=' * 40}")
    print(f"📊 {dataset_name}")
    print(f"{'=' * 40}")
    print(f"MAE  (Средняя абсолютная ошибка):  {mae:,.0f} руб")
    print(f"     → В среднем модель ошибается на {mae:,.0f} руб")
    print(f"\nMSE  (Средняя квадратичная ошибка): {mse:,.0f}")
    print(f"     → Штраф за большие ошибки: {mse:,.0f}")
    print(f"\nRMSE (Корень из MSE):              {rmse:,.0f} руб")
    print(f"     → Типичная ошибка модели: ±{rmse:,.0f} руб")
    print(f"\nR²   (Коэффициент детерминации):   {r2:.3f}")
    if r2 > 0.7:
        print(f"     → ✅ Хороший результат! Модель объясняет {r2 * 100:.1f}% вариации цены")
    elif r2 > 0.4:
        print(f"     → 📊 Средний результат. Модель объясняет {r2 * 100:.1f}% вариации цены")
    else:
        print(f"     → ⚠️ Низкий результат. Возможно, нужны дополнительные признаки")

    return mae, mse, rmse, r2


# Рассчитываем метрики для обучающей выборки
train_mae, train_mse, train_rmse, train_r2 = calculate_metrics(y_train, y_train_pred, "ОБУЧАЮЩАЯ ВЫБОРКА (TRAIN)")

# Рассчитываем метрики для тестовой выборки
test_mae, test_mse, test_rmse, test_r2 = calculate_metrics(y_test, y_test_pred, "ТЕСТОВАЯ ВЫБОРКА (TEST)")

# Анализируем разницу между train и test (признак переобучения)
print(f"\n{'=' * 40}")
print(f"🔍 ДИАГНОСТИКА ПЕРЕОБУЧЕНИЯ")
print(f"{'=' * 40}")

diff_r2 = train_r2 - test_r2
print(f"Разница R² (Train - Test): {diff_r2:.3f}")

if diff_r2 < 0:
    print(f"✅ Отлично! Модель на тесте работает даже лучше, чем на обучении")
    print(f"   (маловероятно для реальных данных, возможно, повезло с разбиением)")
elif diff_r2 < 0.05:
    print(f"✅ Хорошо! Модель не переобучена (разница менее 0.05)")
elif diff_r2 < 0.1:
    print(f"⚠️ Есть небольшое переобучение (разница {diff_r2:.3f})")
    print(f"   Можно попробовать добавить регуляризацию")
else:
    print(f"❌ Сильное переобучение! (разница {diff_r2:.3f})")
    print(f"   Модель запомнила обучающие данные, но не умеет обобщать")
    print(f"   Решение: уменьшить сложность модели или добавить регуляризацию")

print("=" * 60)

# ==============================================================================
# БЛОК 10: ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ
# ==============================================================================

"""
ГРАФИКИ ДЛЯ АНАЛИЗА КАЧЕСТВА МОДЕЛИ:

1. Предсказание vs Истина (Scatter plot)
   - Идеальная модель: все точки на красной линии y=x
   - Разброс вокруг линии = ошибка модели

2. Остатки (Residuals) = y_true - y_pred
   - Должны быть случайно разбросаны вокруг 0
   - Систематический тренд = модель не уловила какую-то закономерность
   - Нормальное распределение остатков = хорошо
"""

print("\n📊 ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ")
print("-" * 40)

# Создаём большой график (2 строки, 2 колонки)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# ===== ГРАФИК 1: Train - Предсказание vs Истина =====
ax1 = axes[0, 0]
ax1.scatter(y_train, y_train_pred, alpha=0.5, c='blue', label='Train')
# Идеальная линия (y_pred = y_true)
min_val = min(y_train.min(), y_test.min())
max_val = max(y_train.max(), y_test.max())
ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Идеал (y=x)')
ax1.set_xlabel('Истинная цена (руб)', fontsize=12)
ax1.set_ylabel('Предсказанная цена (руб)', fontsize=12)
ax1.set_title(f'Обучающая выборка (Train)\nR² = {train_r2:.3f}', fontsize=14)
ax1.legend()
ax1.grid(True, alpha=0.3)

# ===== ГРАФИК 2: Test - Предсказание vs Истина =====
ax2 = axes[0, 1]
ax2.scatter(y_test, y_test_pred, alpha=0.5, c='red', label='Test')
ax2.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Идеал (y=x)')
ax2.set_xlabel('Истинная цена (руб)', fontsize=12)
ax2.set_ylabel('Предсказанная цена (руб)', fontsize=12)
ax2.set_title(f'Тестовая выборка (Test)\nR² = {test_r2:.3f}', fontsize=14)
ax2.legend()
ax2.grid(True, alpha=0.3)

# ===== ГРАФИК 3: Остатки на обучающей выборке =====
# Остатки = разница между истиной и предсказанием
residuals_train = y_train - y_train_pred
ax3 = axes[1, 0]
ax3.scatter(y_train_pred, residuals_train, alpha=0.5, c='blue')
ax3.axhline(y=0, color='r', linestyle='--', lw=2)
ax3.set_xlabel('Предсказанная цена (руб)', fontsize=12)
ax3.set_ylabel('Остатки (руб)', fontsize=12)
ax3.set_title('Остатки на обучающей выборке', fontsize=14)
ax3.grid(True, alpha=0.3)

# ===== ГРАФИК 4: Распределение остатков (гистограмма) =====
residuals_test = y_test - y_test_pred
ax4 = axes[1, 1]
ax4.hist(residuals_test, bins=40, edgecolor='black', alpha=0.7, color='red', density=True)
# Добавляем нормальное распределение для сравнения
from scipy.stats import norm

mu, std = norm.fit(residuals_test)
x = np.linspace(residuals_test.min(), residuals_test.max(), 100)
ax4.plot(x, norm.pdf(x, mu, std), 'k-', lw=2, label=f'Норм. распред. (μ={mu:.0f}, σ={std:.0f})')
ax4.axvline(x=0, color='g', linestyle='--', lw=2, label='Ноль (идеал)')
ax4.set_xlabel('Остатки (руб)', fontsize=12)
ax4.set_ylabel('Плотность', fontsize=12)
ax4.set_title('Распределение остатков на тесте\n(должно быть похоже на нормальное)', fontsize=14)
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Дополнительный анализ остатков
print("\n📊 АНАЛИЗ ОСТАТКОВ")
print("-" * 40)
print(f"Среднее остатков (должно быть ≈0): {residuals_test.mean():.0f} руб")
print(f"Стандартное отклонение остатков: {residuals_test.std():.0f} руб")
print(f"Медиана остатков: {residuals_test.median():.0f} руб")

# Проверка на систематическую ошибку
if abs(residuals_test.mean()) < 1000:
    print("✅ Остатки симметричны относительно 0 (систематической ошибки нет)")
else:
    print("⚠️ Остатки смещены — модель систематически завышает/занижает предсказания")

print("=" * 60)

# ==============================================================================
# БЛОК 11: ИНТЕРПРЕТАЦИЯ КОЭФФИЦИЕНТОВ (ДЛЯ CATEGORICAL)
# ==============================================================================

"""
ЧТО ДЕЛАЮТ ВЕСА В ЛИНЕЙНОЙ РЕГРЕССИИ?
- Положительный вес: увеличение признака → увеличение предсказания
- Отрицательный вес: увеличение признака → уменьшение предсказания

ВАЖНО: Для OneHotEncoder веса интерпретируются как "средний вклад категории"
по сравнению с базовой (которая закодирована как все нули)
"""

print("\n🔍 ИНТЕРПРЕТАЦИЯ КОЭФФИЦИЕНТОВ")
print("-" * 40)

# Получаем названия всех признаков после OneHotEncoder
# Это важно для понимания, какой вес к чему относится
feature_names = []

# Добавляем названия числовых признаков
feature_names.extend(numeric_features)

# Добавляем названия категориальных признаков (после OneHot)
for i, cat_feature in enumerate(categorical_features):
    encoder = trained_preprocessor.named_transformers_['categorical']
    categories = encoder.categories_[i]
    for category in categories:
        feature_names.append(f"{cat_feature}_{category}")

# Создаём DataFrame с весами для удобного просмотра
coefficients_df = pd.DataFrame({
    'Признак': feature_names,
    'Вес (коэффициент)': trained_model.coef_
})

# Сортируем по абсолютному значению веса (чтобы увидеть самые важные признаки)
coefficients_df['|Вес|'] = np.abs(coefficients_df['Вес (коэффициент)'])
coefficients_df = coefficients_df.sort_values('|Вес|', ascending=False).drop('|Вес|', axis=1)

print("\n📊 ВЕСА ПРИЗНАКОВ (по убыванию важности):")
print(coefficients_df.to_string(index=False))
