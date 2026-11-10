"""Step 3: interactive 2-page dashboard (single HTML, Plotly.js embedded, works offline).
Slicers: Card tier, Income band, Gender, Age band, Risk band. All KPIs/charts recompute on filter change."""
import pandas as pd, json
from plotly.offline import get_plotlyjs
df=pd.read_csv("BankChurners_cleaned.csv")
cols=["Churned","Card_Category","Income_Category","Gender","Age_Band","Risk_Band","Trans_Freq_Category","Total_Relationship_Count",
      "Contacts_Count_12_mon","Months_Inactive_12_mon","Value_Class","Credit_Limit","Avg_Utilization_Ratio","Total_Trans_Amt","Total_Trans_Ct"]
data=df[cols].round(3).to_dict("records")
opts={c:(["All"]+sorted(df[c].astype(str).unique(),key=lambda x:x)) for c in ["Card_Category","Income_Category","Gender","Age_Band","Risk_Band"]}
opts["Age_Band"]=["All","<35","35-44","45-54","55-64","65+"];opts["Risk_Band"]=["All","Low","Medium","High"]
opts["Card_Category"]=["All","Blue","Silver","Gold","Platinum"]
opts["Income_Category"]=["All","Less than $40K","$40K - $60K","$60K - $80K","$80K - $120K","$120K +","Not Disclosed"]
html=f"""<!DOCTYPE html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Bank Customer Retention Dashboard</title><script>{get_plotlyjs()}</script>
<style>:root{{--navy:#0B3556;--red:#C8372D;--bg:#fff;--card:#EEF3F8;--tx:#222}}@media(prefers-color-scheme:dark){{:root{{--bg:#14181d;--card:#1f2630;--tx:#e8edf2;--navy:#8fc1ec}}}}
body{{font-family:Segoe UI,Arial,sans-serif;margin:0;background:var(--bg);color:var(--tx)}}header{{background:#0B3556;color:#fff;padding:14px 22px}}header h1{{margin:0;font-size:20px}}header p{{margin:4px 0 0;font-size:12px;opacity:.85}}
.bar{{display:flex;gap:14px;flex-wrap:wrap;padding:10px 22px;border-bottom:1px solid #8884;align-items:end}}.bar label{{font-size:11px;display:block;margin-bottom:2px;font-weight:600}}select{{padding:5px 8px;font-size:13px}}
button.tab{{padding:8px 16px;border:0;background:var(--card);color:var(--tx);font-weight:600;cursor:pointer;border-radius:6px 6px 0 0}}button.tab.on{{background:var(--navy);color:#fff}}
.kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:10px;padding:14px 22px}}.k{{background:var(--card);border-left:4px solid var(--red);padding:10px;border-radius:6px}}.k b{{font-size:22px;color:var(--navy);display:block}}.k span{{font-size:11px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:12px;padding:0 22px 22px}}.grid>div{{height:300px;min-width:0}}.note{{padding:0 22px 14px;font-size:12px;opacity:.85}}</style></head><body>
<header><h1>Customer Retention &amp; Performance Intelligence - Retail Card Portfolio</h1><p>Source: BankChurners, Goyal (n.d.), Kaggle. n = 10,127. Retention-risk band is a rule-based score (see report).</p></header>
<div class="bar" id="bar"></div>
<div class="bar" style="border:0;padding-bottom:0"><button class="tab on" id="t1" onclick="show(1)">Page 1: Executive Overview</button><button class="tab" id="t2" onclick="show(2)">Page 2: Retention Analysis</button></div>
<div class="kpis" id="kpis"></div>
<div id="p1" class="grid"><div id="c1"></div><div id="c2"></div><div id="c3"></div><div id="c4"></div><div id="c5"></div></div>
<div id="p2" class="grid" style="display:none"><div id="d1"></div><div id="d2"></div><div id="d3"></div><div id="d4"></div><div id="d5"></div></div>
<p class="note">Red bars are more than 1 percentage point above the filtered-population average (dashed line). Small groups (n &lt; 30) are greyed and labelled to avoid over-reading noise.</p>
<script>
const D={json.dumps(data,separators=(",",":"))};const O={json.dumps(opts)};const F={{}};
const bar=document.getElementById("bar");const names={{Card_Category:"Card tier",Income_Category:"Income band",Gender:"Gender",Age_Band:"Age band",Risk_Band:"Risk band"}};
for(const k in O){{const d=document.createElement("div");d.innerHTML=`<label>${{names[k]}}</label><select onchange="F['${{k}}']=this.value;draw()">${{O[k].map(v=>`<option>${{v}}</option>`).join("")}}</select>`;bar.appendChild(d);F[k]="All"}}
function show(n){{p1.style.display=n==1?"grid":"none";p2.style.display=n==2?"grid":"none";t1.className="tab"+(n==1?" on":"");t2.className="tab"+(n==2?" on":"");draw()}}
const mean=a=>a.length?a.reduce((x,y)=>x+y,0)/a.length:0;
function rates(rows,key,order){{const m={{}};rows.forEach(r=>{{(m[r[key]]=m[r[key]]||[]).push(r.Churned)}});const ks=order||Object.keys(m).sort();return ks.filter(k=>m[k]).map(k=>({{k:String(k),n:m[k].length,r:mean(m[k])*100}}))}}
function bars(id,rows,key,title,order,avg){{const t=rates(rows,key,order);Plotly.react(id,[{{type:"bar",x:t.map(o=>o.k),y:t.map(o=>o.r),text:t.map(o=>o.r.toFixed(1)+"%<br>n="+o.n),textposition:"outside",marker:{{color:t.map(o=>o.n<30?"#bbb":(o.r>avg+1?"#C8372D":"#B8C4CF"))}},hovertemplate:"%{{x}}: %{{y:.1f}}% attrition<extra></extra>"}}],
{{title:{{text:title,font:{{size:13}}}},margin:{{t:40,b:50,l:45,r:10}},yaxis:{{title:"Attrition %",rangemode:"tozero"}},shapes:[{{type:"line",xref:"paper",x0:0,x1:1,y0:avg,y1:avg,line:{{dash:"dash",color:"#0B3556"}}}}],paper_bgcolor:"rgba(0,0,0,0)",plot_bgcolor:"rgba(0,0,0,0)",font:{{color:getComputedStyle(document.body).color}}}},{{displayModeBar:false,responsive:true}})}}
function draw(){{const R=D.filter(r=>Object.keys(F).every(k=>F[k]=="All"||String(r[k])==F[k]));const n=R.length;const avg=mean(R.map(r=>r.Churned))*100;
const K=[["Customers",n.toLocaleString()],["Attrition rate",avg.toFixed(1)+"%"],["Active ratio",(mean(R.map(r=>r.Months_Inactive_12_mon<=2?1:0))*100).toFixed(1)+"%"],["Avg credit limit","$"+Math.round(mean(R.map(r=>r.Credit_Limit))).toLocaleString()],["Avg utilisation",(mean(R.map(r=>r.Avg_Utilization_Ratio))*100).toFixed(1)+"%"],["Avg products",mean(R.map(r=>r.Total_Relationship_Count)).toFixed(2)],["High-risk share",(mean(R.map(r=>r.Risk_Band=="High"?1:0))*100).toFixed(1)+"%"]];
kpis.innerHTML=K.map(x=>`<div class="k"><b>${{x[1]}}</b><span>${{x[0]}}</span></div>`).join("");if(!n)return;
if(p1.style.display!="none"){{bars("c1",R,"Card_Category","Attrition by card tier",["Blue","Silver","Gold","Platinum"],avg);bars("c2",R,"Income_Category","Attrition by income band",O.Income_Category.slice(1),avg);bars("c3",R,"Age_Band","Attrition by age band",O.Age_Band.slice(1),avg);bars("c4",R,"Risk_Band","Actual attrition by risk band",["Low","Medium","High"],avg);bars("c5",R,"Value_Class","Attrition by customer value class",["Low Value","Mid Value","High Value"],avg)}}
else{{bars("d1",R,"Trans_Freq_Category","Attrition by yearly transaction count",["Very Low (<=40)","Low (41-60)","Medium (61-80)","High (>80)"],avg);bars("d2",R,"Total_Relationship_Count","Attrition by products held",null,avg);bars("d3",R,"Contacts_Count_12_mon","Attrition by bank contacts (12 mo)",null,avg);
const x=["Very Low (<=40)","Low (41-60)","Medium (61-80)","High (>80)"],y=[1,2,3,4,5,6];const z=x.map(a=>y.map(b=>{{const s=R.filter(r=>r.Trans_Freq_Category==a&&r.Total_Relationship_Count==b);return s.length>=10?+(mean(s.map(r=>r.Churned))*100).toFixed(0):null}}));
Plotly.react("d4",[{{type:"heatmap",x:y.map(v=>v+" products"),y:x,z:z,colorscale:"Reds",text:z.map(r=>r.map(v=>v==null?"":v+"%")),texttemplate:"%{{text}}",hovertemplate:"%{{y}}, %{{x}}: %{{z}}%<extra></extra>"}}],{{title:{{text:"Heatmap: attrition % (activity x products)",font:{{size:13}}}},margin:{{t:40,b:40,l:110,r:10}},paper_bgcolor:"rgba(0,0,0,0)",font:{{color:getComputedStyle(document.body).color}}}},{{displayModeBar:false,responsive:true}});
const s=R.length>2500?R.filter((_,i)=>i%Math.ceil(R.length/2500)==0):R;const tr=(f,c,nm)=>({{type:"scattergl",mode:"markers",name:nm,x:s.filter(r=>r.Churned==f).map(r=>r.Total_Trans_Ct),y:s.filter(r=>r.Churned==f).map(r=>r.Total_Trans_Amt),marker:{{size:4,opacity:.55,color:c}}}});
Plotly.react("d5",[tr(0,"#B8C4CF","Existing"),tr(1,"#C8372D","Attrited")],{{title:{{text:"Scatter: transactions vs spend",font:{{size:13}}}},xaxis:{{title:"Transactions"}},yaxis:{{title:"Amount ($)"}},margin:{{t:40,b:45,l:55,r:10}},paper_bgcolor:"rgba(0,0,0,0)",plot_bgcolor:"rgba(0,0,0,0)",font:{{color:getComputedStyle(document.body).color}}}},{{displayModeBar:false,responsive:true}})}}}}
draw();
</script></body></html>"""
open("Bank_Retention_Dashboard_Interactive.html","w").write(html);print(len(html)/1e6,"MB")
