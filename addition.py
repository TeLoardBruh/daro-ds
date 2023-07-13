import pandas as pd
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

data = [
{
        "InvoiceDate":  "08-07-2023",
        "CompanyName":   "L001-901502790",
        "InvoiceNumber": "0200003841",
        "Currency":  "Riel(រ)",
        "TotalPrice":   11070,
        "itemPurchase": "Ice Shaken Hibicus Lemon Tea",
        "Status":   "Success",
        "LuckyDrawID":  "000000548761",
        "Additional": ""
        },

{
        "InvoiceDate":  "30-06-2023",
        "CompanyName":   "ហេង ស្រីពេជ្រ",
        "InvoiceNumber": "N/A",
        "Currency":  "Dollar($)",
        "TotalPrice":   40,
        "itemPurchase": "Tent",
        "Status":   "Success",
        "LuckyDrawID":  "000000618761",
        "Additional": ""
        },

{
        "InvoiceDate":  "25-06-2023",
        "CompanyName":   "E009-2100001928",
        "InvoiceNumber": "DD0235RC202306/02428",
        "Currency":  "Riel(រ)",
        "TotalPrice":   9020,
        "itemPurchase": "Other Bakery",
        "Status":   "Success",
        "LuckyDrawID":  "000000533310",
        "Additional": ""
        },
{
        "InvoiceDate":  "03-06-2023",
        "CompanyName":   "L001-902006483",
        "InvoiceNumber": "0000073973",
        "Currency":  "Dollar($)",
        "TotalPrice":   6.20,
        "itemPurchase":   "Dutch Mill Drink",
        "Status":   "Success",
        "LuckyDrawID":  "000000500523",
        "Additional": ""
        },
{
        "InvoiceDate":  "23-04-2023",
        "CompanyName":   "សួនទឹកកំពង់ស្ពឺ",
        "InvoiceNumber": "134",
        "Currency":  "Riel(រ)",
        "TotalPrice":   20000,
        "itemPurchase": "Meatball (L)",
        "Status":   "Success",
        "LuckyDrawID":  "000000487151",
        "Additional": ""
        },
{
        "InvoiceDate":  "09-04-2023",
        "CompanyName":   "Tressco Cafe 3",
        "InvoiceNumber": "R2304006400",
        "Currency":  "Riel(រ)",
        "TotalPrice":   30000,
        "itemPurchase": "Hot Capuccino",
        "Status":   "Success",
        "LuckyDrawID":  "000000487195",
        "Additional": ""
        },
{
        "InvoiceDate":  "18-03-2023",
        "CompanyName":   "TUBE COFFEE STM",
        "InvoiceNumber": "0334",
        "Currency":  "Riel(រ)",
        "TotalPrice":   19598,
        "itemPurchase": "Aarabica Capucinno",
        "Status":   "Success",
        "LuckyDrawID":  "000000487163",
        "Additional": ""
        },
{
        "InvoiceDate":  "21-03-2023",
        "CompanyName":   "អាហារដ្ធាន&កាហ្វេឆាផះយ៉ម (Chapayom)",
        "InvoiceNumber": "1",
        "Currency":  "Riel(រ)",
        "TotalPrice":   19000,
        "itemPurchase": "Seafood Spegetti",
        "Status":   "Success",
        "LuckyDrawID":  "000000487813",
        "Additional": ""
        },
{
        "InvoiceDate":  "30-05-2023",
        "CompanyName":   "E105-1900001122",
        "InvoiceNumber": "0000271",
        "Currency":  "Riel(រ)",
        "TotalPrice":   668700,
        "itemPurchase":   "Toothpaste",
        "Status":   "Success",
        "LuckyDrawID":  "000000484018",
        "Additional": ""
        },
{
        "InvoiceDate":  "16-05-2023",
        "CompanyName":   "TARZAN",
        "InvoiceNumber": "1",
        "Currency":  "Riel(រ)",
        "TotalPrice":   39000,
        "itemPurchase":   "Fish Fillet with tomato lemon dish",
        "Status":   "Success",
        "LuckyDrawID":  "000000462284",
        "Additional": ""
        },
{
        "InvoiceDate":  "08-05-2023",
        "CompanyName":   "L001-107016060",
        "InvoiceNumber": "010010170098",
        "Currency":  "Dollar($)",
        "TotalPrice":   13.30,
        "itemPurchase":   "Vasaline cream",
        "Status":   "Success",
        "LuckyDrawID":  "000000461822",
        "Additional": ""
        },

{
        "InvoiceDate":  "29-04-2023",
        "CompanyName":   "Bangkok Papaya Salad",
        "InvoiceNumber": "N/A",
        "Currency":  "Riel(រ)",
        "TotalPrice":   42000,
        "itemPurchase":   "Beef rice",
        "Status":   "Success",
        "LuckyDrawID":  "000000461716",
        "Additional": ""
        },
{
        "InvoiceDate":  "15-05-2023",
        "CompanyName":   "LANDO SUPPER MART",
        "InvoiceNumber": "LS-031626",
        "Currency":  "Riel(រ)",
        "itemPurchase":    "Body dendorant",
        "TotalPrice":   51875,
        "Status":   "Success",
        "LuckyDrawID":  "000000461685",
        "Additional": ""
        },
{
        "InvoiceDate":  "08-02-2023",
        "CompanyName":   "Cafe Amazon",
        "InvoiceNumber": "DD0235RC202302/00947",
        "Currency":  "Riel(រ)",
        "TotalPrice":   8815,
        "itemPurchase":   "Ice Latte Amazon",
        "Status":   "Success",
        "LuckyDrawID":  "000000320959",
        "Additional": ""
        },
{
        "InvoiceDate":  "07-02-2023",
        "CompanyName":   "Cafe Amazon",
        "InvoiceNumber": "DD0235RC2023/00939",
        "Currency":  "Riel(រ)",
        "TotalPrice":   17630,
        "itemPurchase":   "Green tea latte",
        "Status":   "Success",
        "LuckyDrawID":  "000000320003",
        "Additional": ""
        },
{
        "InvoiceDate":  "24-01-2023",
        "CompanyName":   "L001-104008962",
        "InvoiceNumber": "B005-23-008587",
        "Currency":  "Riel(រ)",
        "TotalPrice":   43200,
        "Status":   "Success",
        "itemPurchase":  "Ice Coffee Latte",
        "LuckyDrawID":  "000000296836",
        "Additional": ""
        },
{
        "InvoiceDate":  "15-01-2023",
        "CompanyName":   "B Best Coffee",
        "InvoiceNumber": "Order-47",
        "Currency":  "Riel(រ)",
        "TotalPrice":   28249,
        "itemPurchase":   "Ice coffee latte",
        "Status":   "Success",
        "LuckyDrawID":  "000000284991",
        "Additional": ""
        },
{
        "InvoiceDate":  "14-01-2023",
        "CompanyName":   "Cafe Amazon",
        "InvoiceNumber": "DD023RC202301/01444",
        "Currency":  "Riel(រ)",
        "TotalPrice":   17630,
        "itemPurchase":   "Ice Strawberry milk",
        "Status":   "Success",
        "LuckyDrawID":  "000000284535",
        "Additional": ""
        },
{
        "InvoiceDate":  "06-01-2023",
        "CompanyName":   "DEY TMEY IN STATION CAFE AMAZON",
        "InvoiceNumber": "DD0235RC202301/00616",
        "Currency":  "Riel(រ)",
        "TotalPrice":   26445,
        "itemPurchase":   "blueberry milk coffee",
        "Status":   "Success",
        "LuckyDrawID":  "000000280522",
        "Additional": ""
        },
{
        "InvoiceDate":  "01-01-2023",
        "CompanyName":   "L001-902006483",
        "InvoiceNumber": "0000029214",
        "Currency":  "Riel(រ)",
        "TotalPrice":   14000,
        "itemPurchase": "Hot dog cheese",
        "Status":   "Success",
        "LuckyDrawID":  "000000280526",
        "Additional": ""
        },
{
        "InvoiceDate":  "05-01-2023",
        "CompanyName":   "Cafe Amazon Kompong Speu",
        "InvoiceNumber": "DD0066RC202301/00865",
        "Currency":  "Riel(រ)",
        "TotalPrice":   8815,
        "Status":   "Success",
        "itemPurchase": "Ice Black Coffee",
        "LuckyDrawID":  "000000280382",
        "Additional": ""
        },
{
        "InvoiceDate":  "01-01-2023",
        "CompanyName":   "Cafe Amazon Kompong Speu",
        "InvoiceNumber": "DD0066RC202301/00126",
        "Currency":  "Riel(រ)",
        "TotalPrice":   42435,
        "itemPurchase": "Ice Lemon Tea",
        "Status":   "Success",
        "LuckyDrawID":  "000000280395",
        "Additional": ""
        },
{
        "InvoiceDate":  "31-12-2022",
        "CompanyName":   "E009-1900000647",
        "InvoiceNumber": "PP1-IN00035339",
        "Currency":  "Riel(រ)",
        "TotalPrice":   94620,
        "Status":   "Success",
        "itemPurchase":  "Toilet Cleansing Liquid",
        "LuckyDrawID":  "000000280413",
        "Additional": ""
        },
{
        "InvoiceDate":  "05-01-2023",
        "CompanyName":   "L001-100181805",
        "InvoiceNumber": "0454012300000273",
        "Currency":  "Riel(រ)",
        "TotalPrice":   79200,
        "itemPurchase":  "banana",
        "Status":   "Success",
        "LuckyDrawID":  "000000280440",
        "Additional": ""
        },
{
        "InvoiceDate":  "01-01-2023",
        "CompanyName":   "E009-1900000647",
        "InvoiceNumber": "PP1-IN00035654",
        "Currency":  "Riel(រ)",
        "TotalPrice":   29424,
        "Status":   "Success",
        "itemPurchase": "Cherry",
        "LuckyDrawID":  "000000280460",
        "Additional": ""
        },
{
        "InvoiceDate":  "12-01-2023",
        "CompanyName":   "ម្អម",
        "InvoiceNumber": "17",
        "Currency":  "Riel(រ)",
        "TotalPrice":   72000,
        "itemPurchase": "BBQ Grill Set",
        "Status":   "Success",
        "LuckyDrawID":  "000000280461",
        "Additional": ""
        },
{
        "InvoiceDate":  "11-01-2023",
        "CompanyName":   "B105-902100906",
        "InvoiceNumber": "KCM01000073870",
        "Currency":  "Riel(រ)",
        "TotalPrice":   16399,
        "Status":   "Success",
        "itemPurchase": "Gum",
        "LuckyDrawID":  "000000278470",
        "Additional": ""
        },
{
        "InvoiceDate":  "12-01-2023",
        "CompanyName":   "BRAND INT COLLECTION",
        "InvoiceNumber": "KPSIN0000937",
        "Currency":  "Riel(រ)",
        "TotalPrice":   58384,
        "itemPurchase":  "Fluffy toys",
        "Status":   "Success",
        "LuckyDrawID":  "000000280475",
        "Additional": ""
        },
{
        "InvoiceDate":  "15-11-2022",
        "CompanyName":   "E105-1900001122",
        "InvoiceNumber": "0000020",
        "Currency":  "Riel(រ)",
        "TotalPrice":   20600,
        "itemPurchase": "Flipflop",
        "Status":   "Success",
        "LuckyDrawID":  "000000269951",
        "Additional": ""
        },
{
        "InvoiceDate":  "17-12-2022",
        "CompanyName":   "TRESSCO CAFE 3",
        "InvoiceNumber": "R2212042772",
        "Currency":  "Riel(រ)",
        "TotalPrice":   128000,
        "itemPurchase": "hotpot",
        "Status":   "Success",
        "LuckyDrawID":  "000000257404",
        "Additional": ""
        },
{
        "InvoiceDate":  "26-12-2022",
        "CompanyName":   "TRESSCO OMPE PHNOM",
        "InvoiceNumber": "R2212076101",
        "Currency":  "Riel(រ)",
        "TotalPrice":   14000,
        "itemPurchase": "Omelette Rice",
        "Status":   "Success",
        "LuckyDrawID":  "000000256496",
        "Additional": ""
        },
{
        "InvoiceDate":  "24-12-2022",
        "CompanyName":   "PTTCL-Cafe Amazon Kampong Speu Station",
        "InvoiceNumber": "DD0066RC202212/03838",
        "Currency":  "Riel(រ)",
        "TotalPrice":   27060,
        "itemPurchase": "Ice Latte",
        "Status":   "Success",
        "LuckyDrawID":  "000000256214",
        "Additional": ""
        },
{
        "InvoiceDate":  "24-12-2022",
        "CompanyName":   "Mr.LY",
        "InvoiceNumber": "000247",
        "Currency":  "Dollar($)",
        "TotalPrice":   540,
        "itemPurchase": "Car brake equipment",
        "Status":   "Success",
        "LuckyDrawID":  "000000256218",
        "Additional": ""
        },
{
        "InvoiceDate":  "15-12-2022",
        "CompanyName":   "ម្អម",
        "InvoiceNumber": "MV04",
        "Currency":  "Riel(រ)",
        "TotalPrice":   8000,
        "itemPurchase": "Noodle soup",
        "Status":   "Success",
        "LuckyDrawID":  "000000249634",
        "Additional": ""
        },
{
        "InvoiceDate":  "12-01-2023",
        "CompanyName":   "ស្ថានីយ៏ទឹកស្អាតព្រៃផ្តៅ",
        "InvoiceNumber": "N/A",
        "Currency":  "Riel(រ)",
        "TotalPrice":   30900,
        "itemPurchase": "Car brake equipment",
        "Status":   "Success",
        "LuckyDrawID":  "000000244514",
        "Additional": ""
        },
{
        "InvoiceDate":  "13-12-2022",
        "CompanyName":   "CAFE AMAZON KOMPONG SPEU",
        "InvoiceNumber": "DD0066RC202212/02209",
        "Currency":  "Riel(រ)",
        "TotalPrice":   15990,
        "itemPurchase": "Passion smoothie",
        "Status":   "Success",
        "LuckyDrawID":  "000000239777",
        "Additional": ""
        },
{
        "InvoiceDate":  "10-12-2022",
        "CompanyName":   "B105-902100906",
        "InvoiceNumber": "KCM01000025436",
        "Currency":  "Riel(រ)",
        "TotalPrice":   8220,
        "itemPurchase": "Pringles snack",
        "Status":   "Success",
        "LuckyDrawID":  "000000237298",
        "Additional": ""
        },
{
        "InvoiceDate":  "03-12-2022",
        "CompanyName":   "PTTCL-Dey Tmey in Station Cafe Amazon",
        "InvoiceNumber": "DD235RC202212/00364",
        "Currency":  "Riel(រ)",
        "TotalPrice":   17630,
        "itemPurchase": "Ice Americano",
        "Status":   "Success",
        "LuckyDrawID":  "000000231345",
        "Additional": ""
        },
{
        "InvoiceDate":  "08-12-2022",
        "CompanyName":   "ម្អម",
        "InvoiceNumber": "NA",
        "Currency":  "Riel(រ)",
        "TotalPrice":   52000,
        "itemPurchase": "Smokie tomato beef",
        "Status":   "Success",
        "LuckyDrawID":  "000000236310",
        "Additional": ""
        },
{
        "InvoiceDate":  "25-11-2022",
        "CompanyName":   "LANDO SUPPER MART",
        "InvoiceNumber": "LS-026402",
        "Currency":  "Riel(រ)",
        "TotalPrice":   12658,
        "itemPurchase": "hat",
        "Status":   "Success",
        "LuckyDrawID":  "000000231975",
        "Additional": ""
        },
{
        "InvoiceDate":  "29-10-2022",
        "CompanyName":   "E105-1900001122",
        "InvoiceNumber": "0000017",
        "Currency":  "Riel(រ)",
        "TotalPrice":   61600,
        "itemPurchase": "fruiser handwsah",
        "Status":   "Success",
        "LuckyDrawID":  "000000231979",
        "Additional": ""
        },
{
        "InvoiceDate":  "10-08-2022",
        "CompanyName":   "L001-901802215",
        "InvoiceNumber": "000718653",
        "Currency":  "Riel(រ)",
        "TotalPrice":   2600,
        "itemPurchase": "Transportation service",
        "Status":   "Success",
        "LuckyDrawID":  "000000231339",
        "Additional": ""
        },
{
        "InvoiceDate":  "07-12-2022",
        "CompanyName":   "PTTCL-Dey Tmey in Station Cafe Amazon",
        "InvoiceNumber": "DD0235RC202212/00726",
        "Currency":  "Riel(រ)",
        "TotalPrice":   8815,
        "itemPurchase": "Ice Cafe",
        "Status":   "Success",
        "LuckyDrawID":  "000000231347",
        "Additional": ""
        },
{
        "InvoiceDate":  "29-08-2022",
        "CompanyName":   "B015-902100906",
        "InvoiceNumber": "08048",
        "Currency":  "Riel(រ)",
        "TotalPrice":   16400,
        "itemPurchase": "Ice Capuccino",
        "Status":   "Success",
        "LuckyDrawID":  "000000138142",
        "Additional": ""
        },
{
        "InvoiceDate":  "15-09-2022",
        "CompanyName":   "PPTTCL-Kampong Speu Station cafe amazon",
        "InvoiceNumber": "DD0066RC202209/02820",
        "Currency":  "Dollar($)",
        "TotalPrice":   4.15,
        "itemPurchase": "Green tea frappe",
        "Status":   "Success",
        "LuckyDrawID":  "000000135529",
        "Additional": ""
        },
{
        "InvoiceDate":  "25-07-2022",
        "CompanyName":   "មន្ទីរពេទ្យ កាល់ម៉ែត្រ",
        "InvoiceNumber": "2200347863",
        "Currency":  "Riel(រ)",
        "TotalPrice":   541000,
        "itemPurchase": "Paracetamol",
        "Status":   "Success",
        "LuckyDrawID":  "000000132095",
        "Additional": ""
        },
{
        "InvoiceDate":  "05-09-2022",
        "CompanyName":   "មន្ទីរពេទ្យ កាល់ម៉ែត្រ",
        "InvoiceNumber": "2200434785",
        "Currency":  "Riel(រ)",
        "TotalPrice":   32900,
        "itemPurchase": "painkiller medicine",
        "Status":   "Success",
        "LuckyDrawID":  "000000132001",
        "Additional": ""
        },
{
        "InvoiceDate":  "11-09-2022",
        "CompanyName":   "oktchm",
        "InvoiceNumber": "A57220911-32",
        "Currency":  "Riel(រ)",
        "TotalPrice":   48000,
        "itemPurchase": "Meatball",
        "Status":   "Success",
        "LuckyDrawID":  "000000132049",
        "Additional": ""
        },
{
        "InvoiceDate":  "11-09-2022",
        "CompanyName":   "ឧី អេស អាយ ឡាក់គី ប្រាយវេត លីមីធីត",
        "InvoiceNumber": "0214042200064871",
        "Currency":  "Riel(រ)",
        "TotalPrice":   69200,
        "itemPurchase": "Baby diapers",
        "Status":   "Success",
        "LuckyDrawID":  "000000132066",
        "Additional": ""
        },
{
        "InvoiceDate":  "08-09-2022",
        "CompanyName":   "PTTCL-Kampong Speu Station cafe amazon",
        "InvoiceNumber": "DD0066RC202209/01425",
        "Currency":  "Riel(រ)",
        "TotalPrice":   17015,
        "Status":   "Success",
        "itemPurchase": "Ice amerciano",
        "LuckyDrawID":  "000000127490",
        "Additional": ""
        },
{
        "InvoiceDate":  "03-09-2022",
        "CompanyName":   "Tamago",
        "InvoiceNumber": "0000024926",
        "Currency":  "Dollar($)",
        "TotalPrice":   5,
        "itemPurchase": "Udon hotpot",
        "Status":   "Success",
        "LuckyDrawID":  "000000120565",
        "Additional": ""
        },
{
        "InvoiceDate":  "03-09-2022",
        "CompanyName":   "Amazon PTTCL-Dey Tmey in station",
        "InvoiceNumber": "DD0235RC202209/00258",
        "Currency":  "Riel(រ)",
        "TotalPrice":   16195,
        "itemPurchase": "Black coffee",
        "Status":   "Success",
        "LuckyDrawID":  "000000120653",
        "Additional": ""
        },
{
        "InvoiceDate":  "03-09-2022",
        "CompanyName":   "Amazon PTTCL-Dey Tmey in station",
        "InvoiceNumber": "DD0235RC202209/00243",
        "Currency":  "Riel(រ)",
        "TotalPrice":   24805,
        "itemPurchase": "Extra Large fresh lemonade",
        "Status":   "Success",
        "LuckyDrawID":  "000000119784",
        "Additional": ""
        },
{
        "InvoiceDate":  "27-08-2022",
        "CompanyName":   "L001-901502790",
        "InvoiceNumber": "DD0235RC202209/00258",
        "Currency":  "Riel(រ)",
        "TotalPrice":   6150,
        "itemPurchase": "ice latte",
        "Status":   "Success",
        "LuckyDrawID":  "000000119691",
        "Additional": ""
        },
{
        "InvoiceDate":  "02-09-2022",
        "CompanyName":   "Bonjour",
        "InvoiceNumber": "78759",
        "Currency":  "Riel(រ)",
        "TotalPrice":   33150,
        "itemPurchase": "instant ramen",
        "Status":   "Success",
        "LuckyDrawID":  "000000119714",
        "Additional": ""
        },
{
        "InvoiceDate":  "02-09-2022",
        "CompanyName":   "មាតុភូមិ ក្រុងច្បារមន ខេត្តកំពង់ស្ពឺ",
        "InvoiceNumber": "B0014387",
        "Currency":  "Riel(រ)",
        "TotalPrice":   27000,
        "itemPurchase": "lobster fried rice",
        "Status":   "Success",
        "LuckyDrawID":  "000000118743",
        "Additional": ""
        }
]



dataset = pd.read_csv('convertcsv.csv')
df = pd.DataFrame(dataset)


datasetTrain = pd.read_csv('train_data.csv')

#========================================== Data Cleansing Sectiong ===================================================
# Function to detect and remove duplicate invoice numbers
def detect_duplicate_invoices(df):
    duplicates = df[df.duplicated(subset="InvoiceNumber", keep=False)]
    if not duplicates.empty:
        print("Duplicate invoices detected:")
        df.drop_duplicates(subset="InvoiceNumber", keep=False, inplace=True)
        print("Duplicate invoices dropped.")
    else:
        print("No duplicate invoices found.")


# Data cleaning algorithm
# Convert InvoiceDate column to datetime format
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], format="%d-%m-%Y")
# Remove any leading/trailing whitespaces from string columns
df["CompanyName"] = df["CompanyName"].str.strip()
df["InvoiceNumber"] = df["InvoiceNumber"].str.strip()
df["Currency"] = df["Currency"].str.strip()
df["itemPurchase"] = df["itemPurchase"].str.strip()
df["Status"] = df["Status"].str.strip()

# Replace "N/A" with NaN in InvoiceNumber column
df["InvoiceNumber"].replace("N/A", pd.NA, inplace=True)

# Call the function to detect and remove duplicate invoices
detect_duplicate_invoices(df)

# Print the cleaned DataFrame
# Remove leading and trailing whitespace from the columns
datasetTrain['sale_item'] = datasetTrain['sale_item'].str.strip()
datasetTrain['business_industry'] = datasetTrain['business_industry'].str.strip()

# Remove any duplicate rows
datasetTrain = datasetTrain.drop_duplicates()

# Drop any rows with missing or null values
datasetTrain = datasetTrain.dropna()

# Convert the 'business_industry' column to lowercase
datasetTrain['business_industry'] = datasetTrain['business_industry'].str.lower()

# Remove any leading and trailing whitespace from the 'business_industry' column
datasetTrain['business_industry'] = datasetTrain['business_industry'].str.strip()

# Save the cleaned dataset to a new CSV file
datasetTrain.to_csv('cleaned_data.csv', index=False)

#=========================================== End of Cleansing =========================================================

dataClean = pd.read_csv('cleaned_data.csv')

# Split the data into features (X) and labels (y)
X = dataClean['sale_item']
y = dataClean['business_industry']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

vectorizer = TfidfVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train_vectorized, y_train)

# Evaluate the model
accuracy = model.score(X_test_vectorized, y_test)
print('Accuracy:', accuracy)

# Predict industry for new item descriptions
new_items = ['smoke honey beef dish']  # Replace with your own item descriptions
new_items_vectorized = vectorizer.transform(new_items)
predicted_industries = model.predict(new_items_vectorized)

print(f"Item: {new_items}, Industry: {predicted_industries}")
