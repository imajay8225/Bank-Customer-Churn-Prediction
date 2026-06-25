
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\Users\AJAY PRATAP SINGH\OneDrive\Desktop\Bank_customer_churn_prediction\data\European_Bank.csv")

df.hist(figsize=(12,10))
plt.show()

for col in ["Geography","Gender","NumOfProducts","HasCrCard","IsActiveMember","Exited"]:
    df[col].value_counts().plot(kind="bar",title=col)
    plt.show()

print(df.corr(numeric_only=True))
