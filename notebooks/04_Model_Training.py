
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report,roc_auc_score

df=pd.read_csv(r"C:\Users\AJAY PRATAP SINGH\OneDrive\Desktop\Bank_customer_churn_prediction\data\processed_bank_data.csv")

X=df.drop(columns=["Exited","CustomerId","Surname"],errors="ignore")
y=df["Exited"]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

models={
"Logistic":LogisticRegression(max_iter=1000),
"DecisionTree":DecisionTreeClassifier(random_state=42),
"RandomForest":RandomForestClassifier(random_state=42),
"GradientBoosting":GradientBoostingClassifier(random_state=42)
}

for name,model in models.items():
    model.fit(X_train,y_train)
    pred=model.predict(X_test)
    print(name)
    print(classification_report(y_test,pred))
    if hasattr(model,"predict_proba"):
        print("ROC AUC:",roc_auc_score(y_test,model.predict_proba(X_test)[:,1]))
