# =========================================================
# УРОК 4 (расширенный): УЛУЧШАЕМ МОДЕЛЬ + РАЗБИРАЕМСЯ В ML
# =========================================================
# Цель: на реальном датасете California Housing показать,
# как поэтапно улучшать модель, добавлять признаки, выбирать
# алгоритмы и анализировать ошибки.
#
# Продолжительность: ~2 часа (с объяснениями и паузами)
# =========================================================

# =========================================================
# 0. ИМПОРТЫ
# =========================================================
# Импорт библиотек для работы с данными и визуализации
import pandas as pd                     # работа с таблицами (DataFrame)
import numpy as np                      # работа с массивами чисел, математические функции
import matplotlib.pyplot as plt         # построение графиков
import seaborn as sns                   # продвинутая визуализация, более красивые графики

# Импорт модулей для построения и оценки моделей
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
# train_test_split – разделение данных на обучение и тест
# cross_val_score – кросс-валидация (оценка стабильности модели)
# GridSearchCV – перебор гиперпараметров для поиска лучших

from sklearn.linear_model import LinearRegression, Ridge
# LinearRegression – обычная линейная регрессия (метод наименьших квадратов)
# Ridge – линейная регрессия с L2-регуляризацией

from sklearn.tree import DecisionTreeRegressor  # дерево решений для регрессии
from sklearn.ensemble import RandomForestRegressor  # случайный лес (ансамбль деревьев)

from sklearn.preprocessing import StandardScaler, OneHotEncoder
# StandardScaler – масштабирование признаков (среднее 0, стд 1)
# OneHotEncoder – преобразование категориальных признаков в dummy (0/1)

from sklearn.compose import ColumnTransformer    # позволяет применять разные преобразования к разным колонкам
from sklearn.pipeline import Pipeline            # создаёт конвейер (последовательность шагов)

# Импорт метрик для оценки качества регрессии
from sklearn.metrics import (
    mean_absolute_error,                  # MAE – средняя абсолютная ошибка
    mean_squared_error,                   # MSE – среднеквадратичная ошибка
    mean_absolute_percentage_error,       # MAPE – средняя абсолютная процентная ошибка
    r2_score                              # R² – коэффициент детерминации
)

# =========================================================
# 1. ЗАГРУЗКА И ПЕРВОЕ ЗНАКОМСТВО С ДАННЫМИ
# =========================================================
# Пояснение: датасет California Housing – цены на дома в Калифорнии,
# основан на переписи 1990 года.
# Признаки: расположение, возраст домов, количество комнат и спален,
# население, домохозяйства, доход, близость к океану.
# Целевая переменная: median_house_value (средняя стоимость дома).

# Читаем CSV-файл с данными в DataFrame
df = pd.read_csv('housing.csv')

# Выводим разделитель для удобства чтения консоли
print("=" * 60)
print("1. ПЕРВОЕ ЗНАКОМСТВО С ДАННЫМИ")
print("=" * 60)

# Размер данных: количество строк и столбцов
print("Размер данных (строки, столбцы):", df.shape)

# Первые 5 строк таблицы, чтобы увидеть примеры записей
print("\nПервые 5 строк:")
print(df.head())

# Информация о столбцах: тип данных, количество непустых значений
print("\nТипы данных и пропуски:")
print(df.info())

# Основные статистики для числовых столбцов (среднее, мин, макс, квартили)
print("\nСтатистики числовых признаков:")
print(df.describe().round(2))

# -----------------------------------------------------------
# 📊 Визуализация целевой переменной и связи с доходом
# -----------------------------------------------------------
# Создаём фигуру размером 10x5 дюймов с двумя графиками
plt.figure(figsize=(10, 5))

# Первый график (слева): гистограмма цены
plt.subplot(1, 2, 1)  # 1 строка, 2 столбца, первый график
plt.hist(df['median_house_value'], bins=50, edgecolor='k', alpha=0.7)  # гистограмма
plt.axvline(600000, color='red', linestyle='--', label='Порог очистки (600k)')  # вертикальная линия
plt.xlabel('Цена дома, $')
plt.ylabel('Количество')
plt.title('Распределение median_house_value')
plt.legend()

# Второй график (справа): точечная диаграмма цена vs доход
plt.subplot(1, 2, 2)  # второй график
plt.scatter(df['median_income'], df['median_house_value'], alpha=0.2, s=10)  # точки с прозрачностью
plt.xlabel('Доход домохозяйства (median_income)')
plt.ylabel('Цена дома')
plt.title('Цена vs Доход')
plt.tight_layout()  # компактно размещаем подписи и графики
plt.show()  # показать все графики

# =========================================================
# 2. ПРЕДОБРАБОТКА ДАННЫХ (ОБЯЗАТЕЛЬНО ДО МОДЕЛИРОВАНИЯ)
# =========================================================
print("\n" + "=" * 60)
print("2. ПРЕДОБРАБОТКА")
print("=" * 60)

# ---- 2.1 Пропуски ----
# Модели машинного обучения не умеют работать с NaN (пропусками).
# Причина пропусков: в исходных данных по некоторым домам не указано кол-во спален.
# Заполним медианой – она устойчива к выбросам.
# Сначала посчитаем, сколько пропусков было до заполнения
print(f"Пропусков в total_bedrooms до: {df['total_bedrooms'].isna().sum()}")
# Заменяем все NaN на медианное значение столбца
df['total_bedrooms'] = df['total_bedrooms'].fillna(df['total_bedrooms'].median())
# Проверяем, что пропусков не осталось
print(f"После заполнения: {df['total_bedrooms'].isna().sum()}")

# ---- 2.2 Обработка категориального признака ----
# ocean_proximity – текстовая категория (близость к океану).
# Превращаем её в числа с помощью One-Hot Encoding (dummy-переменные).
# drop_first=True – удаляем первую категорию, чтобы избежать мультиколлинеарности.
df = pd.get_dummies(df, columns=['ocean_proximity'], drop_first=True)
# Показываем, какие новые колонки добавились
print("\nДобавлены dummy-признаки океана:")
print([col for col in df.columns if 'ocean_proximity' in col])

# ---- 2.3 Удаление явных выбросов ----
# В описании данных сказано, что цены искусственно обрезаны на 500 001,
# но есть и значения до 600 000. Оставим только объекты <= 600000,
# чтобы не искажать модель ультра-дорогими домами.
initial_rows = len(df)  # запоминаем исходное количество строк
df = df[df['median_house_value'] <= 600000]  # отбираем только строки, где цена <= 600000
print(f"Удалено строк: {initial_rows - len(df)}, осталось: {len(df)}")

# =========================================================
# 3. РАЗВЕДОЧНЫЙ АНАЛИЗ (EDA) – КЛЮЧЕВОЙ ЭТАП
# =========================================================
print("\n" + "=" * 60)
print("3. РАЗВЕДОЧНЫЙ АНАЛИЗ (EDA)")
print("=" * 60)

# Корреляционная матрица: посмотрим, какие признаки сильнее всего связаны с ценой.
# Метод .corr() даёт матрицу корреляции всех числовых столбцов.
# Выбираем только строку для 'median_house_value' и сортируем по убыванию.
corr = df.corr()['median_house_value'].sort_values(ascending=False)
print("\nКорреляция признаков с median_house_value:")
print(corr)

# Тепловая карта (heatmap) – наглядное представление корреляций.
# Покажем только верхний треугольник матрицы, чтобы не дублировать информацию.
plt.figure(figsize=(12, 8))
mask = np.triu(np.ones_like(df.corr(), dtype=bool))  # маска для верхнего треугольника
sns.heatmap(df.corr(), mask=mask, annot=True, fmt='.2f', cmap='coolwarm', center=0)
plt.title('Матрица корреляций')
plt.tight_layout()
plt.show()

# =========================================================
# 4. BASELINE МОДЕЛЬ (ТОЛЬКО ОДИН ПРИЗНАК)
# =========================================================
print("\n" + "=" * 60)
print("4. BASELINE (НАИВНЫЙ ПОДХОД)")
print("=" * 60)

# Выбираем самый сильный по корреляции признак – median_income.
# X_bl – признаки (features), y – целевая переменная.
X_bl = df[['median_income']]  # двойные скобки, чтобы получить DataFrame, а не Series
y = df['median_house_value']

# Делим данные на обучающую (80%) и тестовую (20%) выборки.
# random_state=42 – фиксируем случайность для воспроизводимости.
X_train, X_test, y_train, y_test = train_test_split(
    X_bl, y, test_size=0.2, random_state=42
)

# Создаём простую линейную регрессию (baseline)
baseline_model = LinearRegression()
baseline_model.fit(X_train, y_train)  # обучаем модель

# Оценим качество на тестовой выборке с помощью R²
baseline_r2 = baseline_model.score(X_test, y_test)
print(f"Baseline R² (только median_income): {baseline_r2:.3f}")

# Кросс-валидация: проверим, насколько стабильна оценка R²,
# если несколько раз случайно делить данные на обучение и тест.
# cv=5 – 5 фолдов; scoring='r2' – метрика R².
cv_scores = cross_val_score(baseline_model, X_bl, y, cv=5, scoring='r2')
print(f"Среднее CV R² baseline: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

# =========================================================
# 5. УНИВЕРСАЛЬНАЯ ФУНКЦИЯ ОЦЕНКИ (DRY-принцип)
# =========================================================
# Чтобы не повторять код для каждой модели, создаём функцию, которая
# будет вычислять метрики и строить диагностические графики.
def evaluate_model(model, X_train, X_test, y_train, y_test, model_name=""):
    """
    Оценивает модель на тренировочном и тестовом наборах,
    выводит несколько метрик и строит график остатков.
    """
    # Предсказания модели на обучающей и тестовой выборках
    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)

    # Вычисляем основные метрики
    r2_train = r2_score(y_train, pred_train)
    r2_test = r2_score(y_test, pred_test)
    mae = mean_absolute_error(y_test, pred_test)
    rmse = np.sqrt(mean_squared_error(y_test, pred_test))
    mape = mean_absolute_percentage_error(y_test, pred_test) * 100  # переводим в проценты

    # Печатаем результаты
    print(f"\n--- {model_name} ---")
    print(f"R² (train): {r2_train:.3f}")
    print(f"R² (test):  {r2_test:.3f}")
    print(f"MAE:  {mae:,.0f} $")
    print(f"RMSE: {rmse:,.0f} $")
    print(f"MAPE: {mape:.1f} %")

    # Визуализация остатков (residuals) – важный диагностический инструмент.
    # Остаток = реальное значение – предсказанное. Если модель хорошая,
    # остатки должны быть распределены симметрично вокруг 0.
    residuals = y_test - pred_test
    plt.figure(figsize=(12, 5))

    # График остатков от предсказанных значений
    plt.subplot(1, 2, 1)
    plt.scatter(pred_test, residuals, alpha=0.3)
    plt.axhline(y=0, color='red', linestyle='--')  # горизонтальная линия на уровне 0
    plt.xlabel('Предсказанная цена')
    plt.ylabel('Остаток (реальная – предсказанная)')
    plt.title(f'Остатки: {model_name}')

    # Гистограмма распределения остатков (проверка на нормальность)
    plt.subplot(1, 2, 2)
    sns.histplot(residuals, bins=50, kde=True)  # kde – сглаженная кривая плотности
    plt.xlabel('Ошибка')
    plt.title('Распределение ошибок')

    plt.tight_layout()
    plt.show()

    # Возвращаем предсказания на тесте, они могут пригодиться дальше
    return pred_test

# =========================================================
# 6. FEATURE ENGINEERING (СОЗДАНИЕ ОСМЫСЛЕННЫХ ПРИЗНАКОВ)
# =========================================================
print("\n" + "=" * 60)
print("5. FEATURE ENGINEERING")
print("=" * 60)

# Исходные признаки total_rooms, total_bedrooms, population – абсолютные величины.
# Более информативны относительные на одно домохозяйство.
# Создаём новые признаки:

# Комнат на одно домохозяйство (просторность жилья)
df['rooms_per_household'] = df['total_rooms'] / df['households']
# Доля спален среди всех комнат (планировка)
df['bedrooms_ratio'] = df['total_bedrooms'] / df['total_rooms']
# Количество жильцов на домохозяйство (населённость)
df['people_per_household'] = df['population'] / df['households']
# Среднее число спален на домохозяйство (вместимость по спальням)
df['bedrooms_per_household'] = df['total_bedrooms'] / df['households']

print("Добавлены инженерные признаки:")
print(" - rooms_per_household")
print(" - bedrooms_ratio")
print(" - people_per_household")
print(" - bedrooms_per_household")

# Собираем список всех признаков, которые будем использовать для обучения.
# Это числовые колонки + новые инженерные + dummy-признаки океана.
ocean_dummy_cols = [col for col in df.columns if 'ocean_proximity' in col]
feature_cols = [
    'median_income',
    'housing_median_age',
    'rooms_per_household',
    'bedrooms_ratio',
    'people_per_household',
    'bedrooms_per_household',
    'longitude',   # долгота (географическое положение)
    'latitude'     # широта
] + ocean_dummy_cols  # добавляем все dummy-колонки

# Формируем матрицу признаков X и вектор целевой переменной y
X = df[feature_cols]
y = df['median_house_value']

# Новое деление на обучающую и тестовую выборки с полным набором признаков
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nКоличество признаков: {X.shape[1]}")

# =========================================================
# 7. МАСШТАБИРОВАНИЕ ПРИЗНАКОВ (ДЛЯ ЛИНЕЙНЫХ МОДЕЛЕЙ)
# =========================================================
# Линейные модели (включая Ridge) чувствительны к масштабу признаков.
# Стандартизация (StandardScaler) приводит каждый признак к нулевому среднему
# и единичному стандартному отклонению. Деревьям масштабирование не нужно.
# Мы используем Pipeline, чтобы масштабирование происходило внутри кросс-валидации.

# =========================================================
# 8. ОБУЧЕНИЕ И СРАВНЕНИЕ МОДЕЛЕЙ
# =========================================================
print("\n" + "=" * 60)
print("6. ОБУЧЕНИЕ МОДЕЛЕЙ")
print("=" * 60)

# --- Линейная регрессия с регуляризацией (Ridge) ---
# Ridge добавляет штраф за большие веса (L2), что помогает бороться
# с переобучением и мультиколлинеарностью.
# Создаём пайплайн: сначала масштабируем, потом применяем Ridge.
pipe_ridge = Pipeline([
    ('scaler', StandardScaler()),        # шаг 1: масштабирование
    ('ridge', Ridge(alpha=1.0))          # шаг 2: Ridge-регрессия
])
pipe_ridge.fit(X_train, y_train)        # обучаем весь пайплайн
# Оцениваем модель с помощью нашей функции
pred_ridge = evaluate_model(pipe_ridge, X_train, X_test, y_train, y_test, "Ridge регрессия")

# --- Дерево решений (простое) ---
# Ограничение max_depth=5 не даёт дереву слишком углубляться и запоминать шум.
tree = DecisionTreeRegressor(max_depth=5, random_state=42)
tree.fit(X_train, y_train)
pred_tree = evaluate_model(tree, X_train, X_test, y_train, y_test, "Дерево решений (max_depth=5)")

# --- Случайный лес ---
# Ансамбль из 100 деревьев, каждое обучается на случайной подвыборке данных и признаков.
# n_jobs=-1 ускоряет обучение, используя все ядра процессора.
forest = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
forest.fit(X_train, y_train)
pred_forest = evaluate_model(forest, X_train, X_test, y_train, y_test, "Случайный лес (100 деревьев)")

# =========================================================
# 9. КРОСС-ВАЛИДАЦИЯ – ОЦЕНКА СТАБИЛЬНОСТИ
# =========================================================
print("\n" + "=" * 60)
print("7. КРОСС-ВАЛИДАЦИЯ")
print("=" * 60)

# Словарь с уже обученными моделями для сравнения
models = {
    "Ridge": pipe_ridge,
    "Decision Tree": tree,
    "Random Forest": forest
}

# Для каждой модели считаем кросс-валидацию на всём наборе X, y (без разделения).
# Это показывает, насколько устойчиво качество.
for name, model in models.items():
    cv_r2 = cross_val_score(model, X, y, cv=5, scoring='r2')
    print(f"{name}: среднее R²={cv_r2.mean():.3f} (±{cv_r2.std():.3f})")

# =========================================================
# 10. ГИПЕРПАРАМЕТРИЧЕСКАЯ НАСТРОЙКА (НА ПРИМЕРЕ СЛУЧАЙНОГО ЛЕСА)
# =========================================================
print("\n" + "=" * 60)
print("8. ПОДБОР ГИПЕРПАРАМЕТРОВ (GridSearchCV)")
print("=" * 60)

# Задаём сетку параметров, которые будем перебирать.
# Полная сетка (закомментирована) включает много комбинаций.
# Для ускорения используем уменьшенный набор (param_grid_small).
param_grid_small = {
    'n_estimators': [100, 200],                # количество деревьев
    'max_depth': [10, 20, None],               # максимальная глубина каждого дерева
    'min_samples_split': [2, 5]                # минимальное число объектов в узле, чтобы делиться дальше
}

# Создаём GridSearchCV: он переберёт все комбинации параметров,
# для каждой выполнит кросс-валидацию (cv=3) и запомнит лучшую.
grid_search = GridSearchCV(
    RandomForestRegressor(random_state=42, n_jobs=-1),  # базовая модель
    param_grid_small,                                   # сетка параметров
    cv=3,                                               # 3 фолда для скорости
    scoring='r2',                                       # метрика для выбора лучшей модели
    verbose=1,                                          # показывать прогресс
    n_jobs=-1                                           # использовать все ядра
)
grid_search.fit(X_train, y_train)   # запускаем поиск

# Выводим лучшие параметры и создаём модель с ними
print("\nЛучшие параметры:", grid_search.best_params_)
best_forest = grid_search.best_estimator_   # это уже готовая обученная модель
# Оцениваем улучшенный лес
pred_best_forest = evaluate_model(best_forest, X_train, X_test, y_train, y_test, "Лучший Random Forest (GridSearch)")

# =========================================================
# 11. СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ
# =========================================================
print("\n" + "=" * 60)
print("9. СРАВНЕНИЕ ВСЕХ МОДЕЛЕЙ")
print("=" * 60)

# Собираем все результаты в один DataFrame для наглядного сравнения.
results = {
    'Model': [
        'Baseline (LR, только доход)',
        'Ridge (много признаков)',
        'Дерево решений',
        'Случайный лес (базовый)',
        'Случайный лес (настроенный)'
    ],
    'R² test': [
        baseline_r2,
        r2_score(y_test, pred_ridge),
        r2_score(y_test, pred_tree),
        r2_score(y_test, pred_forest),
        r2_score(y_test, pred_best_forest)
    ],
    'MAE, $': [
        # Для baseline используем X_test[['median_income']], т.к. он обучался только на одном признаке
        mean_absolute_error(y_test, baseline_model.predict(X_test[['median_income']])),
        mean_absolute_error(y_test, pred_ridge),
        mean_absolute_error(y_test, pred_tree),
        mean_absolute_error(y_test, pred_forest),
        mean_absolute_error(y_test, pred_best_forest)
    ]
}
results_df = pd.DataFrame(results)
# Округляем для красоты: R² до трёх знаков, MAE до целых долларов
print(results_df.round({'R² test': 3, 'MAE, $': 0}))

# =========================================================
# 12. ВИЗУАЛИЗАЦИЯ ПРЕДСКАЗАНИЙ ЛУЧШЕЙ МОДЕЛИ
# =========================================================
print("\n" + "=" * 60)
print("10. ГРАФИК: РЕАЛЬНАЯ vs ПРЕДСКАЗАННАЯ ЦЕНА")
print("=" * 60)

plt.figure(figsize=(8, 6))
# Точечный график: по оси X реальные цены, по Y – предсказанные
plt.scatter(y_test, pred_best_forest, alpha=0.3, label='Предсказания')
# Диагональная линия – идеальное совпадение
plt.plot([0, 600000], [0, 600000], 'r--', label='Идеальная линия')
plt.xlabel('Реальная цена, $')
plt.ylabel('Предсказанная цена, $')
plt.title('Random Forest (настроенный): предсказание vs реальность')
plt.legend()
plt.grid(True, alpha=0.3)  # сетка для удобства
plt.show()

# =========================================================
# 13. АНАЛИЗ ОШИБОК – ГДЕ МОДЕЛЬ ОШИБАЕТСЯ?
# =========================================================
print("\n" + "=" * 60)
print("11. АНАЛИЗ ОШИБОК")
print("=" * 60)

# Считаем абсолютную ошибку для каждого объекта
errors = np.abs(y_test - pred_best_forest)
# Создаём DataFrame с реальной ценой, предсказанной и абсолютной ошибкой
error_df = pd.DataFrame({
    'real': y_test,
    'pred': pred_best_forest,
    'abs_error': errors
}).sort_values('abs_error', ascending=False)  # сортируем по убыванию ошибки

print("ТОП-10 наихудших предсказаний:")
print(error_df.head(10))  # выводим 10 самых больших ошибок

# Чтобы понять причины ошибок, посмотрим характеристики этих объектов
worst_indices = error_df.head(10).index  # индексы объектов с наихудшими ошибками
print("\nХарактеристики объектов с наихудшими предсказаниями:")
# Берём из исходного df только нужные столбцы
display_df = df.loc[worst_indices, [
    'median_income', 'housing_median_age', 'rooms_per_household',
    'bedrooms_ratio', 'people_per_household', 'median_house_value'
] + ocean_dummy_cols]
print(display_df)

# =========================================================
# 14. ВАЖНОСТЬ ПРИЗНАКОВ
# =========================================================
print("\n" + "=" * 60)
print("12. ВАЖНОСТЬ ПРИЗНАКОВ (СЛУЧАЙНЫЙ ЛЕС)")
print("=" * 60)

# Случайный лес может оценить, насколько каждый признак был полезен
# при построении деревьев (feature importances).
importances = best_forest.feature_importances_
# Создаём Series с именами признаков и их важностью, сортируем по убыванию
feat_imp = pd.Series(importances, index=feature_cols).sort_values(ascending=False)

print("Важность признаков:")
print(feat_imp.round(3))

# Визуализируем горизонтальной столбчатой диаграммой
plt.figure(figsize=(10, 6))
sns.barplot(x=feat_imp.values, y=feat_imp.index, palette='viridis')
plt.title('Feature Importances (Random Forest)')
plt.xlabel('Важность')
plt.tight_layout()
plt.show()

# =========================================================
# 15. ВЫВОДЫ И РЕКОМЕНДАЦИИ
# =========================================================
print("\n" + "=" * 60)
print("13. ВЫВОДЫ")
print("=" * 60)

# Печатаем ключевые выводы, которые студенты должны запомнить
print("""
1. Feature Engineering критически важен: новые признаки сильнее коррелируют с ценой.
2. Больше признаков (включая координаты и категорию океана) дали прирост качества.
3. Random Forest значительно обходит линейные модели, потому что зависимости нелинейны.
4. Кросс-валидация подтверждает стабильность результатов.
5. Настройка гиперпараметров (GridSearch) дала дополнительное улучшение.
6. Остатки модели близки к нормальному распределению – модель адекватна.
7. Главные драйверы цены: median_income, географическое положение (ocean_proximity, longitude, latitude),
   возраст домов и размеры домохозяйств.
8. Для улучшения модели можно добавить взаимодействия признаков, внешние данные (школы, преступность),
   или попробовать градиентный бустинг (XGBoost, LightGBM).
""")

print("Урок завершён. Следующий шаг – ансамбли и бустинг!")