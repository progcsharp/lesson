import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
a=pd.read_csv("exel.csv")

print(a.head())
print(a)

print(a.describe().astype(int))

print(a.info())

print(a.iloc[1])
print(a.loc[1])

print(a[a["Ratings(%)"]>5])

a.plot(y=["Ratings(%)"],linestyle="--", linewidth=2, color="orange")
plt.xlabel('Ось x')
plt.ylabel('Ось y')
plt.title("График рейтинга")
plt.grid(True)
plt.show()

a.plot(y=["Change(%)"],linestyle="--", linewidth=2, color="brown")
plt.xlabel('Ось x')
plt.ylabel('Ось y')
plt.title("График изменений")
plt.grid(True)
plt.show()

plt.hist(a["Ratings(%)"],bins=5,color='green', alpha=0.5, edgecolor='black',rwidth=0.8, label=["Процент рейтинга"])
plt.title("Рейтинг в %")
plt.ylabel("Количество пользователей")
plt.xlabel("Значение")
plt.grid(True,alpha=0.3)
plt.show()

plt.hist(a["Change(%)"],bins=None,color='red', alpha=0.5, edgecolor='black',rwidth=0.8)
plt.title("Изменения в %")
plt.ylabel("Количество пользователей")
plt.xlabel("Значение")
plt.grid(True,alpha=0.3)
plt.show()

plt.scatter(a['Jan 2026'], a["Ratings(%)"], c='blue', s=80)
plt.title('Зависимость рейтинга от дней')
plt.xlabel('День')
plt.ylabel('Рейтинг')
plt.grid(True)
plt.show()

plt.scatter(a['Jan 2026'], a["Change(%)"], c='red', s=80)
plt.title('Зависимость изменений от дней')
plt.xlabel('День')
plt.ylabel('Рейтинг')
plt.grid(True)
plt.show()