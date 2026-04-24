import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
np.random.seed(42)

n=50

table={"name":np.random.choice(["Gigabyte","DELL","HP","ASUS","Acer","Lenovo"],size=50),
       "price":np.random.uniform(100,1000, size=n).round(1),
       "age":np.random.randint(1,11,size=n)
       }
df=pd.DataFrame(table)

def price_of_age(row): # Зависимость цены от возраста модели
    a=row['price']
    if row['age'] <= 3:
        a *= 1
    elif row['age'] <= 5:
        a *= 0.9
    elif row['age'] <= 7:
        a *= 0.8
    else:
        a *= 0.7
    return a

df['discount'] = df.apply(price_of_age, axis=1).round(1)
print(df.head())
df.to_csv("price_2.csv",index=False)
b=pd.read_csv("price_2.csv")

x=np.arange(1,n+1)
y=b['age']
print(len(x),len(y))
plt.scatter(x,y,c="blue", marker="o", alpha=0.7)
plt.grid(True,alpha=0.3)
a,b=np.polyfit(x,y,1)
plt.plot(x, a*x + b, color="red")
plt.xlabel("Количество покупателей")
plt.ylabel("Возраст компов")
plt.show()