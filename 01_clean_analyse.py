"""MIT 8339 - Bank churn BI lab. Step 1: profiling, cleaning, feature engineering, KPIs, risk score.
Run: python 01_clean_analyse.py   (expects BankChurners.csv in same folder)
Data: Goyal, S. (n.d.). Credit Card Customers [Data set]. Kaggle."""
import pandas as pd, numpy as np, json

raw = pd.read_csv("BankChurners.csv")
df = raw.copy()

# ---------- BEFORE profile ----------
nb_cols = [c for c in df.columns if c.startswith("Naive_Bayes")]
unk = {c: int((df[c] == "Unknown").sum()) for c in ["Education_Level","Marital_Status","Income_Category"]}
before = {"rows": len(df), "cols": df.shape[1], "true_nulls": int(df.isna().sum().sum()),
          "duplicate_rows": int(df.duplicated().sum()), "duplicate_ids": int(df.CLIENTNUM.duplicated().sum()),
          "unknown_labels": unk, "naive_bayes_cols": len(nb_cols)}

# ---------- Outliers (IQR) - flagged, not removed ----------
num = ["Customer_Age","Months_on_book","Credit_Limit","Total_Trans_Amt","Total_Trans_Ct",
       "Total_Amt_Chng_Q4_Q1","Total_Ct_Chng_Q4_Q1","Avg_Utilization_Ratio"]
outl = {}
for c in num:
    q1,q3 = df[c].quantile([.25,.75]); i = q3-q1
    outl[c] = int(((df[c] < q1-1.5*i) | (df[c] > q3+1.5*i)).sum())
before["iqr_outliers"] = outl

# ---------- Cleaning ----------
df = df.drop(columns=nb_cols)                       # leaked model-output columns
for c in ["Education_Level","Marital_Status","Income_Category"]:
    df[c] = df[c].replace("Unknown", "Not Disclosed")  # keep as own category (no imputation)
df["Attrition_Flag"] = df["Attrition_Flag"].str.strip()
df["Churned"] = (df.Attrition_Flag == "Attrited Customer").astype(int)
df["Gender"] = df.Gender.map({"M":"Male","F":"Female"})
# validity checks
assert (df.Credit_Limit > 0).all() and df.Avg_Utilization_Ratio.between(0,1).all()
chk = (df.Credit_Limit - df.Total_Revolving_Bal - df.Avg_Open_To_Buy).abs().max()
before["open_to_buy_identity_max_gap"] = float(chk)

# ---------- Derived fields ----------
df["Age_Band"] = pd.cut(df.Customer_Age,[0,34,44,54,64,120],labels=["<35","35-44","45-54","55-64","65+"])
df["Utilisation_Category"] = pd.cut(df.Avg_Utilization_Ratio,[-.01,.05,.30,.60,1.0],
                                    labels=["Dormant (<=5%)","Low (5-30%)","Moderate (30-60%)","High (>60%)"])
df["Trans_Freq_Category"] = pd.cut(df.Total_Trans_Ct,[0,40,60,80,200],labels=["Very Low (<=40)","Low (41-60)","Medium (61-80)","High (>80)"])
df["Value_Class"] = pd.qcut(df.Total_Trans_Amt,[0,.25,.75,1],labels=["Low Value","Mid Value","High Value"])
df["Tenure_Band"] = pd.cut(df.Months_on_book,[0,24,36,48,100],labels=["<=24m","25-36m","37-48m","48m+"])
inc_order = ["Less than $40K","$40K - $60K","$60K - $80K","$80K - $120K","$120K +","Not Disclosed"]
df["Income_Category"] = pd.Categorical(df.Income_Category, inc_order)

# ---------- Rule-based retention-risk score (0-100) ----------
# Points assigned from the attrition gaps seen in EDA (transparent, auditable)
s = np.zeros(len(df))
s += np.select([df.Total_Trans_Ct<=40, df.Total_Trans_Ct<=60],[25,12],0)           # low activity
s += np.select([df.Total_Ct_Chng_Q4_Q1<=0.5, df.Total_Ct_Chng_Q4_Q1<=0.7],[20,10],0) # falling txn count
s += np.select([df.Total_Relationship_Count<=2, df.Total_Relationship_Count==3],[15,8],0) # few products
s += np.select([df.Months_Inactive_12_mon>=3, df.Months_Inactive_12_mon==2],[15,6],0)  # inactivity
s += np.select([df.Contacts_Count_12_mon>=4, df.Contacts_Count_12_mon==3],[15,8],0)    # service contacts
s += np.select([df.Avg_Utilization_Ratio<.05, df.Avg_Utilization_Ratio<.15],[10,4],0)  # disengaged revolving use
df["Risk_Score"] = s
df["Risk_Band"] = pd.cut(df.Risk_Score,[-1,29,54,101],labels=["Low","Medium","High"])

df.to_csv("BankChurners_cleaned.csv", index=False)

# ---------- AFTER profile ----------
after = {"rows": len(df), "cols": df.shape[1], "true_nulls": int(df.isna().sum().sum()),
         "unknown_labels": {c:int((df[c]=="Unknown").sum()) for c in ["Education_Level","Marital_Status"]},
         "naive_bayes_cols": 0, "derived_fields": 7}

# ---------- KPIs ----------
kpi = {
 "customers": len(df), "attrition_rate": df.Churned.mean()*100,
 "active_ratio": (df.Months_Inactive_12_mon<=2).mean()*100,
 "avg_credit_limit": df.Credit_Limit.mean(), "avg_utilisation": df.Avg_Utilization_Ratio.mean()*100,
 "avg_trans_amt": df.Total_Trans_Amt.mean(), "avg_products": df.Total_Relationship_Count.mean(),
 "high_value_share": (df.Value_Class=="High Value").mean()*100,
 "high_risk_share": (df.Risk_Band=="High").mean()*100}
print("BEFORE", json.dumps(before, indent=1)); print("AFTER", json.dumps(after, indent=1))
print("KPI", json.dumps({k:round(v,2) for k,v in kpi.items()}, indent=1))

# ---------- Attrited vs existing ----------
cmp_cols = ["Total_Trans_Ct","Total_Trans_Amt","Total_Ct_Chng_Q4_Q1","Total_Amt_Chng_Q4_Q1","Total_Relationship_Count",
            "Months_Inactive_12_mon","Contacts_Count_12_mon","Credit_Limit","Total_Revolving_Bal","Avg_Utilization_Ratio","Customer_Age","Months_on_book"]
cmp = df.groupby("Attrition_Flag")[cmp_cols].mean().T.round(3); cmp["gap_%"]=((cmp["Attrited Customer"]/cmp["Existing Customer"]-1)*100).round(1)
print(cmp)
from scipy import stats
for c in ["Total_Trans_Ct","Total_Trans_Amt","Total_Relationship_Count","Months_Inactive_12_mon","Contacts_Count_12_mon","Credit_Limit","Avg_Utilization_Ratio"]:
    a=df[df.Churned==1][c];b=df[df.Churned==0][c]; u=stats.mannwhitneyu(a,b); print(c,"MWU p=%.2e"%u.pvalue)
def rate(col): 
    t=df.groupby(col,observed=True).Churned.agg(["mean","size"]); t["mean"]=(t["mean"]*100).round(1); return t
for c in ["Card_Category","Income_Category","Gender","Age_Band","Utilisation_Category","Trans_Freq_Category","Value_Class",
          "Total_Relationship_Count","Months_Inactive_12_mon","Contacts_Count_12_mon","Tenure_Band","Risk_Band","Education_Level","Marital_Status"]:
    print("\n",c); print(rate(c))
print(pd.crosstab(df.Risk_Band, df.Churned))
# risk-band precision/recall
hi=df[df.Risk_Band=="High"]; print("HIGH recall",hi.Churned.sum()/df.Churned.sum(),"precision",hi.Churned.mean())
mh=df[df.Risk_Band!="Low"]; print("MED+HIGH recall",mh.Churned.sum()/df.Churned.sum(),"precision",mh.Churned.mean(), "share",len(mh)/len(df))
print("Revenue-at-risk proxy: credit limit of attrited", df[df.Churned==1].Credit_Limit.sum(), "txn amt lost", df[df.Churned==1].Total_Trans_Amt.sum(), "share of txn", df[df.Churned==1].Total_Trans_Amt.sum()/df.Total_Trans_Amt.sum())
json.dump({"before":before,"after":after,"kpi":kpi},open("profile.json","w"),indent=1,default=str)
