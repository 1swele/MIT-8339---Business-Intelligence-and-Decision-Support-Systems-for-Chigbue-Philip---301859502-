"""Author: Onaivi Collins Dania | o.dania5318@miva.edu.ng | Student ID 301855318

MIT 8339 - Business Intelligence and Decision Support Systems (Virtual Lab)
Script 2 of 2: Exploratory analysis, KPI development and risk-score validation (Task 3)

Input : outputs/BankChurners_cleaned.csv  (produced by 01_data_preparation.py)
Output: outputs/kpi_summary.csv, outputs/segment_kpis.csv,
        outputs/attrited_vs_existing.csv, outputs/risk_band_summary.csv,
        outputs/risk_validation.txt, outputs/dashboard_data.json,
        figures/*.png (report charts)

Run   : python 02_analysis_kpis.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

OUT, FIG = "outputs", "figures"
os.makedirs(FIG, exist_ok=True)
df = pd.read_csv(f"{OUT}/BankChurners_cleaned.csv")
N = len(df)

# ---------------------------------------------------------------- 1. HEADLINE KPIs
kpis = {
    "Total customers": N,
    "Attrited customers": int(df["Attrited"].sum()),
    "Attrition rate (%)": round(df["Attrited"].mean() * 100, 2),
    "Active customer ratio (%) [inactive <=2 of last 12 months]": round(df["Active_Customer_Flag"].mean() * 100, 2),
    "Average credit limit ($)": round(df["Credit_Limit"].mean(), 0),
    "Average utilisation ratio (%)": round(df["Avg_Utilization_Ratio"].mean() * 100, 2),
    "Total transaction amount, 12m ($)": int(df["Total_Trans_Amt"].sum()),
    "Average transactions per customer, 12m": round(df["Total_Trans_Ct"].mean(), 1),
    "Average product relationship count": round(df["Total_Relationship_Count"].mean(), 2),
    "Customers with usage decline Q4 vs Q1 (%)": round(df["Engagement_Decline_Flag"].mean() * 100, 2),
    "High retention-risk customers": int((df["Risk_Band"] == "High").sum()),
    "Transaction value lost with attrited customers ($)": int(df.loc[df.Attrited == 1, "Total_Trans_Amt"].sum()),
}
kpi_df = pd.DataFrame({"KPI": list(kpis), "Value": [f"{v:,}" if isinstance(v, int) else f"{v:,.2f}" for v in kpis.values()]})
kpi_df.to_csv(f"{OUT}/kpi_summary.csv", index=False)
print(kpi_df.to_string(index=False))

# ---------------------------------------------------------------- 2. SEGMENT KPIs
seg_cols = ["Gender", "Age_Band", "Income_Segment", "Card_Category", "Education_Level", "Marital_Status",
            "Product_Depth", "Utilisation_Category", "Txn_Frequency_Category", "Customer_Value_Class",
            "Inactivity_Band", "Contact_Band", "Tenure_Band", "Risk_Band"]
seg_rows = []
for c in seg_cols:
    g = df.groupby(c).agg(customers=("CLIENTNUM", "count"), attrited=("Attrited", "sum"),
                          attrition_rate=("Attrited", "mean"), avg_credit_limit=("Credit_Limit", "mean"),
                          avg_utilisation=("Avg_Utilization_Ratio", "mean"), avg_trans_amt=("Total_Trans_Amt", "mean"),
                          avg_trans_ct=("Total_Trans_Ct", "mean"), avg_products=("Total_Relationship_Count", "mean"))
    g["share_of_attriters_%"] = g["attrited"] / df["Attrited"].sum() * 100
    g["lift_vs_avg"] = g["attrition_rate"] / df["Attrited"].mean()
    g = g.reset_index().rename(columns={c: "segment"})
    g.insert(0, "dimension", c)
    seg_rows.append(g)
seg = pd.concat(seg_rows)
seg["attrition_rate"] = (seg["attrition_rate"] * 100).round(1)
seg = seg.round(2)
seg.to_csv(f"{OUT}/segment_kpis.csv", index=False)

# ---------------------------------------------------------------- 3. ATTRITED vs EXISTING (>= 4 variables)
comp_vars = ["Total_Trans_Ct", "Total_Trans_Amt", "Total_Ct_Chng_Q4_Q1", "Total_Amt_Chng_Q4_Q1",
             "Total_Revolving_Bal", "Avg_Utilization_Ratio", "Total_Relationship_Count",
             "Months_Inactive_12_mon", "Contacts_Count_12_mon", "Credit_Limit", "Customer_Age", "Months_on_book"]
rows = []
for v in comp_vars:
    a = df.loc[df.Attrited == 1, v]
    e = df.loc[df.Attrited == 0, v]
    p = mannwhitneyu(a, e).pvalue
    rows.append({"variable": v, "attrited_mean": a.mean(), "existing_mean": e.mean(),
                 "attrited_median": a.median(), "existing_median": e.median(),
                 "difference_%": (a.mean() - e.mean()) / e.mean() * 100,
                 "mann_whitney_p": p, "significant_0.05": "Yes" if p < 0.05 else "No"})
comp = pd.DataFrame(rows).round(3)
comp["mann_whitney_p"] = comp["mann_whitney_p"].map(lambda x: "<0.001" if x < 0.001 else f"{x:.3f}")
comp.to_csv(f"{OUT}/attrited_vs_existing.csv", index=False)
print(comp.to_string(index=False))

# ---------------------------------------------------------------- 4. RISK-BAND VALIDATION (30% hold-out)
train, test = train_test_split(df, test_size=0.30, random_state=42, stratify=df["Attrited"])
auc_test = roc_auc_score(test["Attrited"], test["Risk_Score"])
auc_train = roc_auc_score(train["Attrited"], train["Risk_Score"])
rb = df.groupby("Risk_Band").agg(customers=("CLIENTNUM", "count"), attrited=("Attrited", "sum"),
                                 attrition_rate=("Attrited", "mean"), avg_trans_amt=("Total_Trans_Amt", "mean"),
                                 total_trans_amt=("Total_Trans_Amt", "sum")).reindex(["Low", "Medium", "High"])
rb["share_of_customers_%"] = rb["customers"] / N * 100
rb["share_of_attriters_%"] = rb["attrited"] / df["Attrited"].sum() * 100
rb["attrition_rate"] *= 100
rb_test = test.groupby("Risk_Band")["Attrited"].agg(["count", "mean"]).reindex(["Low", "Medium", "High"])
rb = rb.round(1)
rb.to_csv(f"{OUT}/risk_band_summary.csv")
existing_high = df[(df.Risk_Band == "High") & (df.Attrited == 0)]
with open(f"{OUT}/risk_validation.txt", "w") as f:
    f.write(f"Rule-based risk score AUC - train (70%): {auc_train:.3f}; hold-out (30%): {auc_test:.3f}\n")
    f.write("Hold-out attrition rate by band:\n" + (rb_test.assign(mean=lambda x: (x["mean"] * 100).round(1))).to_string() + "\n")
    f.write(f"Existing (not yet attrited) customers in High band: {len(existing_high)}; "
            f"their 12m transaction value: ${existing_high.Total_Trans_Amt.sum():,.0f}\n")
    f.write(f"Existing customers in Medium band: {((df.Risk_Band=='Medium')&(df.Attrited==0)).sum()}\n")
print(open(f"{OUT}/risk_validation.txt").read())
print(rb[["customers", "attrited", "attrition_rate", "share_of_customers_%", "share_of_attriters_%"]].to_string())

# ---------------------------------------------------------------- 5. INTERACTION: product depth x transaction frequency
heat = df.pivot_table(index="Product_Depth", columns="Txn_Frequency_Category", values="Attrited", aggfunc="mean") * 100
heat = heat.reindex(index=["Thin (1-2)", "Moderate (3-4)", "Deep (5-6)"],
                    columns=["Very Low (<=40)", "Low (41-60)", "Medium (61-80)", "High (>80)"]).round(1)
heat.to_csv(f"{OUT}/heatmap_depth_x_frequency.csv")
print(heat)

# ---------------------------------------------------------------- 6. REPORT FIGURES (matplotlib)
C_ATT, C_EX, C_INK = "#C0392B", "#1F4E79", "#333333"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.titleweight": "bold"})
avg = df["Attrited"].mean() * 100


def bar_rate(dim, order, ax, title):
    s = seg[seg.dimension == dim].set_index("segment").reindex(order)
    cols = [C_ATT if r > avg else C_EX for r in s["attrition_rate"]]
    ax.bar(range(len(s)), s["attrition_rate"], color=cols)
    for i, (r, n) in enumerate(zip(s["attrition_rate"], s["customers"])):
        ax.text(i, r + 0.8, f"{r:.1f}%\n(n={n:,})", ha="center", fontsize=8, color=C_INK)
    ax.axhline(avg, ls="--", color="grey", lw=1)
    ax.text(0.99, 0.97, f"- - bank average {avg:.1f}%", transform=ax.transAxes, ha="right", va="top", fontsize=8, color="grey")
    ax.set_xticks(range(len(s)), s.index, fontsize=8)
    ax.set_ylabel("Attrition rate (%)")
    ax.set_title(title, fontsize=10)
    ax.set_ylim(0, max(s["attrition_rate"]) * 1.35)


fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
bar_rate("Txn_Frequency_Category", ["Very Low (<=40)", "Low (41-60)", "Medium (61-80)", "High (>80)"], axes[0, 0], "Attrition by transaction frequency (12m count)")
bar_rate("Product_Depth", ["Thin (1-2)", "Moderate (3-4)", "Deep (5-6)"], axes[0, 1], "Attrition by product relationship depth")
bar_rate("Utilisation_Category", ["Zero (0%)", "Low (0-30%)", "Moderate (30-70%)", "High (>70%)"], axes[1, 0], "Attrition by credit utilisation")
bar_rate("Contact_Band", ["0-1 contacts", "2-3 contacts", "4+ contacts"], axes[1, 1], "Attrition by service contacts (12m)")
fig.tight_layout()
fig.savefig(f"{FIG}/fig1_attrition_drivers.png", dpi=170)
plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 5))
for flag, c, lab in [(0, C_EX, "Existing"), (1, C_ATT, "Attrited")]:
    s = df[df.Attrited == flag]
    ax.scatter(s["Total_Trans_Ct"], s["Total_Trans_Amt"], s=6, alpha=0.35, color=c, label=lab)
ax.set_xlabel("Total transactions, last 12 months (count)")
ax.set_ylabel("Total transaction amount, last 12 months ($)")
ax.set_title("Attrited customers cluster at low transaction volume")
ax.legend(markerscale=3)
fig.tight_layout()
fig.savefig(f"{FIG}/fig2_scatter_txn.png", dpi=170)
plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 3.6))
im = ax.imshow(heat.values, cmap="Reds", vmin=0, vmax=100)
for i in range(heat.shape[0]):
    for j in range(heat.shape[1]):
        v = heat.values[i, j]
        ax.text(j, i, f"{v:.1f}%", ha="center", va="center", color="white" if v > 50 else C_INK, fontsize=9)
ax.set_xticks(range(heat.shape[1]), heat.columns, fontsize=8)
ax.set_yticks(range(heat.shape[0]), heat.index, fontsize=8)
ax.set_title("Attrition rate: product depth x transaction frequency")
fig.colorbar(im, ax=ax, label="Attrition %")
fig.tight_layout()
fig.savefig(f"{FIG}/fig3_heatmap.png", dpi=170)
plt.close(fig)

# ---------------------------------------------------------------- 7. DASHBOARD DATA (compact record set)
keep = ["CLIENTNUM", "Attrited", "Gender", "Age_Band", "Income_Segment", "Card_Category", "Marital_Status",
        "Education_Level", "Product_Depth", "Utilisation_Category", "Txn_Frequency_Category",
        "Customer_Value_Class", "Inactivity_Band", "Contact_Band", "Risk_Band", "Risk_Score", "Risk_Drivers",
        "Credit_Limit", "Avg_Utilization_Ratio", "Total_Trans_Amt", "Total_Trans_Ct",
        "Total_Relationship_Count", "Active_Customer_Flag", "Engagement_Decline_Flag", "Total_Revolving_Bal",
        "Months_Inactive_12_mon", "Contacts_Count_12_mon", "Total_Ct_Chng_Q4_Q1"]
cols = keep
data = {"columns": cols, "rows": df[cols].values.tolist()}
with open(f"{OUT}/dashboard_data.json", "w") as f:
    json.dump(data, f, separators=(",", ":"), default=lambda o: o.item() if hasattr(o, "item") else str(o))
print("Done. Figures in ./figures, tables in ./outputs")
