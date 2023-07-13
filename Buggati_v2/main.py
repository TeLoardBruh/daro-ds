from function import *
import pandas as pd
import datetime
from pymongo import MongoClient
import csv
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from itertools import islice


client=MongoClient('mongodb://127.0.0.1:27017/')
db = client['Grocery_Store']
ProductCollection = db['Product_list']

Groceries_dataset = pd.read_csv('Groceries_dataset.csv')


df = pd.DataFrame(Groceries_dataset)
# ================================== Cleaning Section ================================== #
Groceries_dataset.itemDescription = Groceries_dataset.itemDescription.str.strip()  #remove whitespace
Groceries_dataset.Date = Groceries_dataset.Date.str.strip()

for date in Groceries_dataset.Date:
    try:
        datetime.strptime(date, '%d-%m-%Y')
    except ValueError:
        print("Abnormal Date format detected !")

for member_number in Groceries_dataset.Member_number:
    if not member_number > 0 and not isinstance(member_number, int):
        print(member_number)



Miquolon = Groceries_dataset['itemDescription']

counterMiq = Counter(Miquolon)
sorted_counts = sorted(counterMiq.items(), key=lambda x: x[1], reverse=True)
i = 0
stringAR =[]
countAR = []
# for string, count in sorted_counts:
#     print(f"string: {string}: {count}")


# plt.bar(stringAR, countAR)
# plt.xlabel('Product Name')
# plt.ylabel('Quantities')
# plt.title('Top 5 Purchase products')
# plt.show()




# ================================== Date Transaction ================================== #
# Determine which day of the week has the most purchase
# beginning of 2014 to end of 2015
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
dateRands = Groceries_dataset['Date']
sorted_dates = sorted(dateRands, key=lambda x: datetime.strptime(x, "%d-%m-%Y"))
CountedDate = Counter(sorted_dates)
start_date = "01-01-2014"
dateThreshold = "01-01-2015"
end_date = "31-12-2015"
format_date = "%d-%m-%Y"
start_annual14 = datetime.strptime(start_date, format_date)
end_annual14 = datetime.strptime(dateThreshold, format_date)
end_annual15 = datetime.strptime(end_date, format_date)
Array2014 = []
Array2015 = []



    # while datetime.strptime(element, format_date) >= end_date:
    #     TotalTransaction20195 = sum(counting)
# plt.plot(TotalTransaction2014, TotalTransaction2015)
# plt.xlabel('2014')
# plt.ylabel('2015')
# plt.title('Comparison')
# print(plt.show())
#================================ END DT ================================================#


#================================ Top 10 items changes over last year =================================================


# Find top 10 customer that bought most of the item in this store
# Find Which item sold the most

filtered_df2014 = df[(df['Date'] >= start_annual14) & (df['Date'] < end_annual14)] # = df['12-31-2014']
filtered_df2015 = df[(df['Date'] >= end_annual14) & (df['Date'] < end_annual15)]

itemSold14 = filtered_df2014['itemDescription']
itemSoldCounted14 = Counter(itemSold14)
iSCSorted14 = sorted(itemSoldCounted14.items(), key=lambda x: x[1], reverse=True)

#print(f" {top5(iSCSorted14)}")


itemSold15 = filtered_df2015['itemDescription']
itemSoldCounted15 = Counter(itemSold15)
iSCSorted15 = sorted(itemSoldCounted15.items(), key=lambda x: x[1], reverse=True)

# ========================= Convert list to Dictionary =======================================
dictConvertKey = []
dictConvertValue = []
for key, value in iSCSorted14:
    dictConvertKey.append(key)
    dictConvertValue.append(value)

result = dict(zip(dictConvertKey, dictConvertValue))

dictConvertKey15 = []
dictConvertValue15 = []
for key15, value15 in iSCSorted15:
    dictConvertKey15.append(key15)
    dictConvertValue15.append(value15)
result15 = dict(zip(dictConvertKey15, dictConvertValue15))

# ============================= end of Conversion ============================================

#============================== Comparison top 5 items =======================================
top5String = []
for kiy in islice(result, 5):
    top5String.append(kiy)


#=============================================================================================
# def moron(top5StringTest, resultTest, result15Test):
#     for snowy in top5StringTest:
#         plt.plot(snowy, resultTest[snowy], linestyle='-', marker='o', label='2014')
#         plt.plot(snowy, result15[snowy], linestyle='-', marker='o', label='2015')
#
#     plt.xlabel('X-axis')
#     plt.ylabel('Y-axis')
#     plt.title('Line Chart')
#
#     plt.legend()
#     plt.show()
#
# moron(top5String, result, result15)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

TotalTransaction2015 = sum(Array2015)
TotalTransaction2014 = sum(Array2014)

values1 = []
values2 = []
for snowy in top5String:
    values1.append(result[snowy])
    values2.append(result15[snowy])

labels = top5String

x = np.arange(len(labels))
width = 0.25
fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, values1, width, label='2014 Stats')
rects2 = ax.bar(x + width/2, values2, width, label='2015 Stats')
ax.bar_label(rects1, padding=3)
ax.bar_label(rects2, padding=3)

ax.set_xlabel('Categories')
ax.set_ylabel('Values')
ax.set_title('Comparison of Dict 1 and Dict 2')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()



plt.show()