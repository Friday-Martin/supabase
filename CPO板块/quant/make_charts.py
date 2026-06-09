"""生成图表(英文标签避免CJK字体缺失)。"""
import os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tickflow import TickFlow

OUT="CPO板块/charts"; os.makedirs(OUT,exist_ok=True)
df=pd.read_csv("CPO板块/quant/aligned_daily.csv",index_col=0,parse_dates=True)

# 图1：收益分解（核心图）
fig,ax=plt.subplots(figsize=(9,5))
ax.plot(df.index,(1+df["cn_full"]).cumprod(),label="Buy&Hold (full)",lw=2)
ax.plot(df.index,(1+df["cn_intraday"]).cumprod(),label="Intraday only (open->close)",lw=2)
ax.plot(df.index,(1+df["cn_gap"]).cumprod(),label="Overnight gap only (close->open)",lw=2)
ax.set_yscale("log"); ax.set_title("A-share CPO basket: return decomposition (2024-2026)")
ax.set_ylabel("cumulative growth (log)"); ax.legend(); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(f"{OUT}/01_return_decomposition.png",dpi=130); plt.close()

# 图2：散点 US隔夜 vs A缺口 / A日内
fig,axs=plt.subplots(1,2,figsize=(12,5))
for ax,col,t in [(axs[0],"cn_gap","US overnight -> A-share OPEN GAP"),
                 (axs[1],"cn_intraday","US overnight -> A-share INTRADAY (open->close)")]:
    d=df[["us_overnight",col]].dropna()
    ax.scatter(d["us_overnight"]*100,d[col]*100,s=8,alpha=.4)
    r=d["us_overnight"].corr(d[col])
    b,a=np.polyfit(d["us_overnight"],d[col],1)
    xs=np.linspace(d["us_overnight"].min(),d["us_overnight"].max(),50)
    ax.plot(xs*100,(b*xs+a)*100,"r-",lw=2)
    ax.set_title(f"{t}\nPearson r = {r:+.3f}"); ax.set_xlabel("US CPO overnight %"); ax.set_ylabel("A-share %")
    ax.axhline(0,color="k",lw=.5); ax.axvline(0,color="k",lw=.5); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(f"{OUT}/02_scatter_gap_vs_intraday.png",dpi=130); plt.close()

# 图3：耦合逐年增强
df["yr"]=df.index.year
rows=[]
for yr,g in df.groupby("yr"):
    rows.append((yr,g["us_overnight"].corr(g["cn_gap"]),g["us_overnight"].corr(g["cn_intraday"]),g["cn_intraday"].mean()*100))
yrs=[str(r[0]) for r in rows]
fig,ax=plt.subplots(figsize=(8,5))
x=np.arange(len(yrs)); w=.35
ax.bar(x-w/2,[r[1] for r in rows],w,label="corr(US night, A gap)")
ax.bar(x+w/2,[r[2] for r in rows],w,label="corr(US night, A intraday)")
ax.set_xticks(x); ax.set_xticklabels(yrs); ax.set_ylabel("Pearson r")
ax.set_title("US->A CPO coupling by year (gap coupling strengthening)")
ax.legend(); ax.grid(alpha=.3,axis="y")
fig.tight_layout(); fig.savefig(f"{OUT}/03_coupling_by_year.png",dpi=130); plt.close()
print("charts saved to",OUT, os.listdir(OUT))
