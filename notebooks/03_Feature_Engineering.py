
import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv(r"C:\Users\AJAY PRATAP SINGH\OneDrive\Desktop\Bank_customer_churn_prediction\data\European_Bank.csv")

df["BalanceSalaryRatio"]=df["Balance"]/(df["EstimatedSalary"]+1)
df["ProductDensity"]=df["NumOfProducts"]/(df["Tenure"]+1)
df["EngagementProduct"]=df["IsActiveMember"]*df["NumOfProducts"]
df["AgeTenure"]=df["Age"]*df["Tenure"]
df["BalancePerProduct"]=df["Balance"]/(df["NumOfProducts"]+1)

df=df.drop(columns=["CustomerId","Surname"])

le=LabelEncoder()
for c in ["Geography","Gender"]:
    df[c]=le.fit_transform(df[c])

print(df.head())
