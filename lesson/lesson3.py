# ================================
# 1. Генерация датасета
# ================================


# housing_median_age
# “средний возраст домов”
# total_rooms
# “общее количество комнат”
# total_bedrooms
# “спальни”
# population
# “сколько людей живёт”
# households
# “сколько домохозяйств (семей)”
# median_income
# “Это самый важный признак — уровень дохода людей”
# ocean_proximity
# “насколько близко к океану”
# median_house_value
# “Это то, что мы хотим объяснить и предсказать”

import pandas as pd  # работа с таблицами
import numpy as np   # работа с числами и случайными данными
import matplotlib.pyplot as plt  # построение графиков

np.random.seed(42)  # фиксируем случайность (чтобы у всех был одинаковый результат)

n = 5000  # количество строк (районов)

data = {
    'longitude': np.random.uniform(-124, -114, n),  # случайные координаты (долгота)
    'latitude': np.random.uniform(32, 42, n),       # случайные координаты (широта)

    'housing_median_age': np.random.choice(  # случайный возраст домов
        [10, 15, 20, 25, 30, 35, 40, 45, 50],
        n,
        p=[0.15, 0.15, 0.15, 0.1, 0.1, 0.1, 0.1, 0.05, 0.1]  # вероятности
    ),

    'total_rooms': np.random.randint(200, 5000, n),       # количество комнат
    'total_bedrooms': np.random.randint(50, 1500, n),     # количество спален
    'population': np.random.randint(100, 5000, n),        # население
    'households': np.random.randint(50, 1500, n),         # домохозяйства
    'median_income': np.random.uniform(1, 12, n),         # доход (1–12 → 10k–120k)

    'ocean_proximity': np.random.choice(  # категориальный признак (локация)
        ['<1H OCEAN', 'INLAND', 'NEAR OCEAN', 'NEAR BAY'],
        n,
        p=[0.4, 0.3, 0.2, 0.1]
    ),
}

df = pd.DataFrame(data)  # создаём таблицу из словаря


def generate_price(row):# функция расчёта цены
    base = row['median_income'] * 40000  # цена зависит от дохода

    # корректировка по локации
    if row['ocean_proximity'] == '<1H OCEAN':
        base *= 1.3
    elif row['ocean_proximity'] == 'NEAR OCEAN':
        base *= 1.2
    elif row['ocean_proximity'] == 'NEAR BAY':
        base *= 1.15
    else:
        base *= 0.8

    price = base + np.random.normal(0, 30000)  # добавляем шум (реализм)
    return np.clip(price, 30000, 500000)  # ограничиваем диапазон

df['median_house_value'] = df.apply(generate_price, axis=1).astype(int)
# применяем функцию к каждой строке → получаем цену

# добавляем пропуски (2%)
missing_idx = np.random.choice(df.index, size=int(n * 0.02), replace=False)
df.loc[missing_idx, 'total_bedrooms'] = np.nan  # ставим NaN

# добавляем выбросы (очень дорогие дома)
outlier_idx = np.random.choice(df.index, size=20, replace=False)
df.loc[outlier_idx, 'median_house_value'] = np.random.randint(700000, 900000, 20)

df.to_csv('housing.csv', index=False)  # сохраняем в CSV
print("✅ Датасет создан")

# ================================
# 2. ЗНАКОМСТВО С ДАННЫМИ (шаг 1)
# ================================

df = pd.read_csv('housing.csv') # загружаем данные

print("=" * 50)
print("ШАГ 1: Знакомство с данными")
print("=" * 50)

print(f"Размер: {df.shape}")# (строки, столбцы)
print(f"\nТипы данных:")
print(df.dtypes) # типы данных
print(f"\nПервые 3 строки:")
print(df.head(3))# первые 3 строки
print(f"\nСтатистика:")
print(df.describe())# базовая статистика

# ================================
# 3. ПОИСК ПРОПУСКОВ (шаг 2)
# ================================

print("\n" + "=" * 50)
print("ШАГ 2: Поиск пропусков")
print("=" * 50)

print(df.isnull().sum()) # считаем пропуски по колонкам

# ✨ ДОПОЛНЕНИЕ 1: показываем процент пропусков
missing_percent = (df['total_bedrooms'].isnull().sum() / len(df)) * 100# считаем процент пропусков
print(f"\nПроцент пропусков в total_bedrooms: {missing_percent:.2f}%")

# ================================
# 4. ОЧИСТКА ДАННЫХ (шаг 3)
# ================================

print("\n" + "=" * 50)
print("ШАГ 3: Очистка данных")
print("=" * 50)

# ✨ ДОПОЛНЕНИЕ 2: сравниваем разные способы очистки
print("Вариант A: Удалить строки с пропусками")
print(f"  Было строк: {len(df)}")
df_dropped = df.dropna(subset=['total_bedrooms'])# удаляем строки с пропусками

print(f"  Стало строк: {len(df_dropped)} (потеряли {len(df) - len(df_dropped)})")

print("\nВариант B: Заполнить медианой (выбираем этот)")
df_clean = df.copy() # копируем данные
df_clean['total_bedrooms'].fillna(df_clean['total_bedrooms'].median(), inplace=True)# заменяем пропуски медианой
print(f"  Пропусков после заполнения: {df_clean['total_bedrooms'].isnull().sum()}")

# ✨ ДОПОЛНЕНИЕ 3: проверка выбросов в цене
print("\nПроверка выбросов в цене:")
print(f"  Минимальная цена: {df_clean['median_house_value'].min()}")
print(f"  Максимальная цена: {df_clean['median_house_value'].max()}")
print(f"  Домов дороже 600k: {(df_clean['median_house_value'] > 600000).sum()}")

# ✨ ДОПОЛНЕНИЕ 4: удаляем выбросы (для чистоты анализа)
df_clean = df_clean[df_clean['median_house_value'] <= 600000]# удаляем выбросы
print(f"\nПосле удаления выбросов (>600k): {len(df_clean)} строк")

# ================================
# 5. ГРУППИРОВКИ И АГРЕГАЦИИ (шаг 4)
# ================================

print("\n" + "=" * 50)
print("ШАГ 4: Группировки и агрегации")
print("=" * 50)

# средняя цена по локации
grouped = df_clean.groupby('ocean_proximity')['median_house_value'].agg(['mean', 'median', 'count'])# группируем по локации и считаем статистику
print("Цены по удалённости от океана:")
print(grouped.sort_values('mean', ascending=False))# сортируем по средней цене


# ✨ ДОПОЛНЕНИЕ 5: создаём категорию возраста домов
def age_category(age):# функция категоризации возраста
    if age < 20:
        return 'Новый'
    elif age < 40:
        return 'Средний'
    else:
        return 'Старый'


df_clean['age_group'] = df_clean['housing_median_age'].apply(age_category)# применяем функцию к колонке
print("\nЦены по возрасту домов:")
print(df_clean.groupby('age_group')['median_house_value'].mean())# средняя цена по возрасту

# ✨ ДОПОЛНЕНИЕ 6: создаём признак "комнат на домохозяйство"
df_clean['rooms_per_household'] = df_clean['total_rooms'] / df_clean['households']# создаём новый признак
print("\nКорреляция 'комнат на домохозяйство' с ценой:")
print(f"  {df_clean['rooms_per_household'].corr(df_clean['median_house_value']):.3f}")# считаем корреляцию

# ================================
# 6. ВИЗУАЛИЗАЦИЯ (шаг 5)
# ================================

print("\n" + "=" * 50)
print("ШАГ 5: Визуализация")
print("=" * 50)

# график 1: доход vs цена # scatter plot (точки: доход vs цена)
plt.figure(figsize=(10, 6))
plt.scatter(df_clean['median_income'], df_clean['median_house_value'], alpha=0.4, s=10)
plt.xlabel('Медианный доход (десятки тысяч $)')
plt.ylabel('Цена дома ($)')
plt.title('Зависимость цены от дохода')
plt.grid(True, alpha=0.3)
plt.show()

# график 2: гистограмма цены # гистограмма распределения
plt.figure(figsize=(10, 6))
plt.hist(df_clean['median_house_value'], bins=50, edgecolor='black', alpha=0.7)
plt.xlabel('Цена дома ($)')
plt.ylabel('Количество районов')
plt.title('Распределение цен на дома')
plt.axvline(df_clean['median_house_value'].median(), color='red', linestyle='--', label='Медиана')
plt.legend()
plt.show()

# ✨ ДОПОЛНЕНИЕ 7: тепловая карта корреляций # матрица корреляций
corr_matrix = df_clean[['median_house_value', 'median_income', 'housing_median_age',
                        'total_rooms', 'population', 'rooms_per_household']].corr()

plt.figure(figsize=(8, 6))
plt.imshow(corr_matrix, cmap='coolwarm', interpolation='none')
plt.colorbar()
plt.xticks(range(len(corr_matrix)), corr_matrix.columns, rotation=45)
plt.yticks(range(len(corr_matrix)), corr_matrix.columns)
for i in range(len(corr_matrix)):
    for j in range(len(corr_matrix)):
        plt.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                 ha='center', va='center',
                 color='white' if abs(corr_matrix.iloc[i, j]) > 0.5 else 'black')
plt.title('Корреляция между признаками')
plt.show()

# ================================
# 7. ВЫВОДЫ (шаг 6 - САМЫЙ ВАЖНЫЙ)
# ================================

print("\n" + "=" * 50)
print("ШАГ 6: ВЫВОДЫ (то, за что платят деньги)")
print("=" * 50)

print("""
📊 **ИТОГИ АНАЛИЗА РЫНКА НЕДВИЖИМОСТИ:**

1. **Главный фактор цены** — медианный доход в районе.
   Корреляция с ценой: {:.2f}

2. **Локация решает всё:**
   - Самые дорогие районы: {} (${:.0f})
   - Самые дешёвые: {} (${:.0f})
   Разница составляет {:.0f}%

3. **Возраст дома** почти не влияет на цену (корреляция {:.2f})

4. **Количество комнат на домохозяйство** — полезный признак
   (корреляция {:.2f})

5. **Аномалии:** {} домов стоят >600k (выбросы/особняки)
""".format(
    df_clean['median_income'].corr(df_clean['median_house_value']),
    grouped.sort_values('mean', ascending=False).index[0],
    grouped.sort_values('mean', ascending=False)['mean'].iloc[0],
    grouped.sort_values('mean', ascending=False).index[-1],
    grouped.sort_values('mean', ascending=False)['mean'].iloc[-1],
    (grouped.sort_values('mean', ascending=False)['mean'].iloc[0] /
     grouped.sort_values('mean', ascending=False)['mean'].iloc[-1] - 1) * 100,
    df_clean['housing_median_age'].corr(df_clean['median_house_value']),
    df_clean['rooms_per_household'].corr(df_clean['median_house_value']),
    (df['median_house_value'] > 600000).sum()
))

# ================================
# 8. [БОНУС/ТИЗЕР] ПРОСТАЯ ML МОДЕЛЬ
# ================================

print("\n" + "=" * 50)
print("БОНУС: Как это превратить в предсказание? (тизер к уроку 5)")
print("=" * 50)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X = df_clean[['median_income']] # признаки (вход)
y = df_clean['median_house_value'] # целевая переменная

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)# делим на обучение и тест

model = LinearRegression() # создаём модель
model.fit(X_train, y_train) # обучаем модель

score = model.score(X_test, y_test)# оцениваем качество (R²)

print(f"Качество модели (R²): {score:.3f}")
print("Это значит, что доход объясняет {:.1f}% изменчивости цены".format(score * 100))

# предсказание для дохода 50,000
sample = [[5]] # пример: доход = 5 (≈ 50k)
prediction = model.predict(sample) # предсказание цены
print(f"\nЕсли доход в районе = $50,000, предсказанная цена дома: ${prediction[0]:.0f}")