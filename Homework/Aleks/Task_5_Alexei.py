import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
n=90

name=["HP", "Asus", "Acer", "Lenovo", "Samsung", "Intel", "Apple", "Dell"]
data={"Название": np.random.choice(name,n),
      "Цена": np.random.uniform(10000,80000,n).round(2),
      "Продажи": np.random.randint(1,10,n)}
data_set = pd.DataFrame(data)
data_set["Дата"]=pd.date_range(start='1/1/2025', periods=len(data_set), freq='D')

new=["Дата", "Название","Цена","Продажи"]
data_set=data_set[new]
# Добавление рейтинга
def rating(row):
    if row=="Dell" or row=="Apple" or row=="HP" :
        return "Отличный"
    elif row=="Lenovo" or row=="Samsung" or row=="Intel":
        return "Нормаьный"
    else:
        return "Так себе"
data_set['Рейтинг'] = data_set["Название"].apply(rating)

# Добавление выручки
data_set["Выручка"]=data_set["Цена"]*data_set["Продажи"]

# Добавление пропусков
missing_idx = np.random.randint(1,90,5)
data_set.loc[missing_idx, 'Цена'] = np.nan

# Добавление выбросов
outlier_idx = np.random.choice(data_set.index, size=5, replace=False)
data_set.loc[outlier_idx, 'Выручка'] = np.random.randint(800000, 900000, 5)

# Сохранение
data_set.to_csv("Comps.csv", index=False)
ds=pd.read_csv("Comps.csv")
# Описание данных
print(ds.shape)
print(ds.dtypes)
print(ds.head(5))
print(ds.describe())
# Замена пропусков медианами
print(ds.isnull().sum())
ds_c=ds.copy()
ds_c["Цена"]=ds_c["Цена"].fillna(ds_c["Цена"].median(), inplace=True)
print(ds_c.isnull().sum())
#
# ds_c=ds_c[ds_c["Выручка"]<800000]
# print(len(ds_c))
# print(ds_c.shape)

print("Сумма общей выручки: ",ds_c["Выручка"].sum().round(2))
print("Средняя выручка: ",ds_c["Выручка"].mean().round(2))
print("Медиана: ",ds_c["Выручка"].median().round(2))
print("Кол-во заказов: ",ds_c["Продажи"].sum())
un, count=np.unique(ds_c["Название"],return_counts=True)
b=count.tolist()
print("Количество уникальных названий: ", dict(zip(un,b)),end="\n\n")

top_day = ds_c.groupby('Дата')['Выручка'].sum().sort_values(ascending=False).head(10)
print("Топ-10 дней по выручке:")
print(top_day,end="\n\n")

rat_rev=ds_c.groupby('Рейтинг')['Цена'].mean().sort_values(ascending=False)
print("Средняя цена взависимости от рейтинга: ")
print(rat_rev.round(2),end="\n\n")

name_sale=ds_c.groupby('Название')["Продажи"].sum().sort_values(ascending=False)
print("Топ товаров по продажам: ")
print(name_sale,end="\n\n")

name_rev=ds_c.groupby('Название')['Выручка'].sum().sort_values(ascending=False)
print("Топ товаров по выручке: ")
print(name_rev,end="\n\n")

name_rat=ds_c.groupby('Рейтинг')["Название"].count().sort_values(ascending=False)
print("Количество товара взависимости от рейтинга: ")
print(name_rat,end="\n\n")

plt.figure(figsize=[8,8])
plt.scatter(ds_c["Цена"],ds_c["Продажи"],alpha=0.5)
plt.title("Цена и продажи")
plt.xlabel("Цена")
plt.ylabel("Количество прожаж")
plt.show()

plt.figure(figsize=[8,8])
plt.plot(ds_c["Продажи"])
plt.title("График количества продаж")
plt.xlabel("Дни")
plt.ylabel("Продажи")
plt.show()

plt.figure(figsize=[10,6])
plt.subplot(1,2,1)
name_rev.plot(kind='bar', color='green',rot=45)
plt.title("Количество выручки по названиям")
plt.grid(axis="y",linestyle="--",alpha=0.5)
plt.xlabel("Названия")
plt.ylabel("Количество выручки")
plt.subplot(1,2,2)
plt.title("Количество продаж по названиям")
name_sale.plot(kind='bar', color='orange', rot=45)
plt.xlabel("Названия")
plt.ylabel("Количество продаж")
plt.grid(axis="y",linestyle="--",alpha=0.5)
plt.show()

plt.figure(figsize=[10,5])
plt.subplot(1,2,1)
name_rat.plot(kind='bar', color='lime',width=0.5,edgecolor='black', rot=0)
plt.grid(linestyle="--",alpha=0.5)
plt.title("Количество продаж по рейтингу")
plt.ylabel("Количество товара")
plt.xlabel("Рейтинг")
plt.subplot(1,2,2)
rat_rev.plot(kind='bar', color='teal',width=0.5,edgecolor='black',rot=0)
plt.grid(linestyle="--",alpha=0.5)
plt.title("Количество выручки по рейтингу")
plt.ylabel("Выручка",color="red")
plt.xlabel("Рейтинг",color="red")
plt.show()

a=ds_c.corr(numeric_only=True)
print(a)
plt.figure(figsize=(10, 8))
plt.matshow(a, cmap="coolwarm")
plt.colorbar()
plt.xticks(range(len(a.columns)), a.columns, rotation="vertical")
plt.yticks(range(len(a.columns)), a.columns)
plt.title("Корреляционная матрица")
plt.show()

