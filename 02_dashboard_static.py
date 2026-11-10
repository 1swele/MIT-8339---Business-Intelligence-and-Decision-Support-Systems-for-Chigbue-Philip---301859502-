"""Step 2: static dashboard page exports (PNG) from the cleaned dataset."""
import pandas as pd, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
df = pd.read_csv("BankChurners_cleaned.csv")
NAVY,RED,GREY,TEAL="#0B3556","#C8372D","#B8C4CF","#2A9D8F"
plt.rcParams.update({"font.family":"DejaVu Sans","axes.spines.top":False,"axes.spines.right":False,"axes.titlesize":10.5,"axes.titleweight":"bold","axes.titlecolor":NAVY,"axes.labelsize":8.5,"xtick.labelsize":8,"ytick.labelsize":8})
avg=df.Churned.mean()*100
def card(fig,x,y,w,h,label,val,sub=""):
    ax=fig.add_axes([x,y,w,h]);ax.axis("off")
    ax.add_patch(FancyBboxPatch((0,0),1,1,boxstyle="round,pad=0,rounding_size=.06",fc="#EEF3F8",ec=NAVY,lw=1,transform=ax.transAxes))
    ax.text(.5,.70,val,ha="center",va="center",fontsize=17,fontweight="bold",color=NAVY);ax.text(.5,.36,label,ha="center",va="center",fontsize=8.5,color="#333")
    if sub: ax.text(.5,.14,sub,ha="center",va="center",fontsize=7,color="#666")
def bar(ax,col,title,order=None,rot=0):
    t=df.groupby(col,observed=True).Churned.mean()*100
    if order is not None:t=t.reindex(order)
    cols=[RED if v>avg+1 else GREY for v in t.values]
    b=ax.bar(range(len(t)),t.values,color=cols);ax.axhline(avg,color=NAVY,ls="--",lw=1);ax.set_xticks(range(len(t)));ax.set_xticklabels([str(i).replace("$",r"\$") for i in t.index],rotation=rot,ha="right" if rot else "center")
    for r,v in zip(b,t.values):ax.text(r.get_x()+r.get_width()/2,v+.5,f"{v:.1f}%",ha="center",fontsize=7.5)
    ax.set_title(title);ax.set_ylabel("Attrition rate (%)");ax.set_ylim(0,max(t.values)*1.18)
# ---------------- PAGE 1 ----------------
fig=plt.figure(figsize=(14,8.6),facecolor="white")
fig.text(.03,.955,"Executive Overview  |  Customer Retention Intelligence - Retail Card Portfolio",fontsize=15,fontweight="bold",color=NAVY)
fig.text(.03,.925,"Source: BankChurners (Goyal, n.d.), n = 10,127. Red bars = more than 1 pt above portfolio average (dashed line = 16.1%). Slicers in the interactive HTML version: Card, Income, Gender, Age Band, Risk Band.",fontsize=8,color="#555")
K=[("Customers",f"{len(df):,}",""),("Attrition Rate",f"{avg:.1f}%",f"{df.Churned.sum():,} attrited"),("Active Customer Ratio",f"{(df.Months_Inactive_12_mon<=2).mean()*100:.1f}%","<=2 inactive months"),
   ("Avg Credit Limit",f"${df.Credit_Limit.mean():,.0f}",""),("Avg Utilisation",f"{df.Avg_Utilization_Ratio.mean()*100:.1f}%",""),("Avg Products / Cust.",f"{df.Total_Relationship_Count.mean():.2f}",""),("High-Risk Share",f"{(df.Risk_Band=='High').mean()*100:.1f}%","of all customers")]
for i,(l,v,s) in enumerate(K): card(fig,.03+i*.1375,.79,.128,.11,l,v,s)
ax=fig.add_axes([.05,.40,.26,.31]);bar(ax,"Card_Category","Attrition by card tier",["Blue","Silver","Gold","Platinum"])
ax=fig.add_axes([.38,.40,.26,.31]);bar(ax,"Income_Category","Attrition by income band",["Less than $40K","$40K - $60K","$60K - $80K","$80K - $120K","$120K +","Not Disclosed"],25)
ax=fig.add_axes([.71,.40,.26,.31]);bar(ax,"Age_Band","Attrition by age band",["<35","35-44","45-54","55-64","65+"])
ax=fig.add_axes([.05,.07,.26,.24]);rb=df.groupby("Risk_Band").agg(n=("Churned","size"),r=("Churned","mean")).reindex(["Low","Medium","High"])
b=ax.bar(rb.index,rb.n,color=[TEAL,"#E9A23B",RED]);ax.set_title("Customers by retention-risk band");ax.set_ylabel("Customers")
for r,n,p in zip(b,rb.n,rb.r):ax.text(r.get_x()+.4,n+60,f"{n:,}\n({p*100:.1f}% churned)",ha="center",fontsize=7.5)
ax.set_ylim(0,rb.n.max()*1.3)
ax=fig.add_axes([.38,.07,.26,.24]);g=df.groupby("Value_Class",observed=True).Churned.mean().reindex(["Low Value","Mid Value","High Value"])*100
b=ax.bar(g.index,g.values,color=[RED if v>avg+1 else GREY for v in g.values]);ax.axhline(avg,color=NAVY,ls="--",lw=1);ax.set_title("Attrition by customer value class");ax.set_ylabel("Attrition rate (%)")
for r,v in zip(b,g.values):ax.text(r.get_x()+.4,v+.5,f"{v:.1f}%",ha="center",fontsize=7.5)
ax.set_ylim(0,32)
ax=fig.add_axes([.71,.05,.26,.28]);ax.axis("off");ax.set_title("Insight box",loc="left")
ax.text(0,.88,"- 1 in 6 customers has left.\n- Risk is behavioural, not demographic:\n  card, income and age gaps are small.\n- Low-value customers churn at 24.4%\n  vs 11.5% for high-value.\n- 1,718 customers (17%) are High Risk;\n  59% of them are already attrited\n  (in-sample validation).",va="top",fontsize=8.5,color="#222",linespacing=1.6)
fig.savefig("dash_page1_executive.png",dpi=130);plt.close(fig)
# ---------------- PAGE 2 ----------------
fig=plt.figure(figsize=(14,8.6),facecolor="white")
fig.text(.03,.955,"Customer Retention Analysis  |  Behavioural drivers of attrition",fontsize=15,fontweight="bold",color=NAVY)
fig.text(.03,.925,"Each chart shows attrition rate by a behavioural dimension; red = more than 1 pt above the 16.1% portfolio average.",fontsize=8,color="#555")
ax=fig.add_axes([.05,.58,.26,.30]);bar(ax,"Trans_Freq_Category","Attrition by yearly transaction count",["Very Low (<=40)","Low (41-60)","Medium (61-80)","High (>80)"],15)
ax=fig.add_axes([.38,.58,.26,.30]);bar(ax,"Total_Relationship_Count","Attrition by number of products held")
ax=fig.add_axes([.71,.58,.26,.30]);bar(ax,"Contacts_Count_12_mon","Attrition by bank contacts (12 mo; 6 contacts n=54)")
ax=fig.add_axes([.10,.08,.21,.38]);pv=df.pivot_table(index="Trans_Freq_Category",columns="Total_Relationship_Count",values="Churned",aggfunc="mean",observed=True).reindex(["Very Low (<=40)","Low (41-60)","Medium (61-80)","High (>80)"])*100
im=ax.imshow(pv.values,cmap="Reds",aspect="auto");ax.set_xticks(range(6));ax.set_xticklabels(pv.columns);ax.set_yticks(range(4));ax.set_yticklabels(pv.index)
for i in range(4):
    for j in range(6):ax.text(j,i,f"{pv.values[i,j]:.0f}%",ha="center",va="center",fontsize=8,color="white" if pv.values[i,j]>40 else "black")
ax.set_xlabel("Products held");ax.set_title("Heatmap: attrition % (activity x products)");ax.spines[:].set_visible(False)
ax=fig.add_axes([.38,.08,.26,.38]);s=df.sample(2500,random_state=1);
for f,c,l in [(0,GREY,"Existing"),(1,RED,"Attrited")]:
    d=s[s.Churned==f];ax.scatter(d.Total_Trans_Ct,d.Total_Trans_Amt,s=6,alpha=.5,color=c,label=l)
ax.set_xlabel("Transactions (12 mo)");ax.set_ylabel("Transaction amount ($)");ax.set_title("Scatter: activity vs spend (2,500 sample)");ax.legend(markerscale=3,fontsize=8)
ax=fig.add_axes([.71,.08,.26,.38]);ax.axis("off");ax.set_title("Priority retention segments (rule-based)",loc="left")
defs=[("<=60 txns AND <=2 products",(df.Total_Trans_Ct<=60)&(df.Total_Relationship_Count<=2)),
("Txn count fell >50% (Q4 vs Q1)",df.Total_Ct_Chng_Q4_Q1<=0.5),
("3+ inactive months AND 3+ contacts",(df.Months_Inactive_12_mon>=3)&(df.Contacts_Count_12_mon>=3)),
("Dormant card (<=5% utilisation)",df.Avg_Utilization_Ratio<=0.05),
("Low-value customers (bottom 25% spend)",df.Value_Class=="Low Value")]
rows=[[n,f"{m.sum():,}",f"{df[m].Churned.mean()*100:.0f}%"] for n,m in defs];seg=pd.DataFrame(rows);print(seg)
t=ax.table(cellText=rows,colLabels=["Behavioural segment","Customers","Attrition"],bbox=[0,.35,1,.6],colWidths=[.58,.22,.20],cellLoc="center");t.auto_set_font_size(False);t.set_fontsize(8.5);
for (r,c),cell in t.get_celld().items():
    if r==0:cell.set_facecolor(NAVY);cell.set_text_props(color="white",fontweight="bold")
fig.savefig("dash_page2_retention.png",dpi=130);plt.close(fig)
print(seg)
