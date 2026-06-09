"""第二轮：把规律挖到可交易粒度。
基于 aligned_daily.csv（cpo_leadlag.py 生成）+ 重新拉个股做横截面。
"""
import os, sys
import numpy as np
import pandas as pd
from tickflow import TickFlow

df = pd.read_csv("CPO板块/quant/aligned_daily.csv", index_col=0, parse_dates=True)
N = len(df)
print("="*78); print(f"样本 N={N}  {df.index.min().date()} ~ {df.index.max().date()}"); print("="*78)

def pr(s): return f"{s*100:+.2f}%"

# ---------- 规律1：缺口越大，日内越容易兑现（高开低走）吗？ ----------
print("\n【规律1】按 A股开盘缺口大小分桶 -> 当日日内(收/开) 表现")
print("  目的：高开是续涨还是兑现？是否存在'缺口越大越要兑现'?")
bins = [-np.inf, -0.03, -0.01, 0, 0.01, 0.03, 0.06, np.inf]
labels = ["<-3%","-3~-1%","-1~0%","0~1%","1~3%","3~6%",">6%"]
df["gap_bucket"] = pd.cut(df["cn_gap"], bins=bins, labels=labels)
g = df.groupby("gap_bucket", observed=True).agg(
    n=("cn_intraday","size"),
    intraday_mean=("cn_intraday","mean"),
    intraday_win=("cn_intraday", lambda s:(s>0).mean()),
    full_mean=("cn_full","mean"),
)
for idx,row in g.iterrows():
    print(f"  缺口{idx:<7} n={int(row['n']):>3}  日内均值 {pr(row['intraday_mean'])}  "
          f"日内胜率 {row['intraday_win']*100:.0f}%  全天均值 {pr(row['full_mean'])}")

# ---------- 规律2：无条件日内漂移 vs 美股条件日内 ----------
print("\n【规律2】日内(收/开)漂移：无条件 vs 美股强涨夜后")
all_in = df["cn_intraday"]
thi = df["us_overnight"].quantile(0.80)
cond_in = df[df["us_overnight"]>=thi]["cn_intraday"]
print(f"  全样本日内: 均值 {pr(all_in.mean())} 胜率 {(all_in>0).mean()*100:.0f}% (n={all_in.notna().sum()})")
print(f"  美股强涨夜日内: 均值 {pr(cond_in.mean())} 胜率 {(cond_in>0).mean()*100:.0f}% (n={len(cond_in)})")
print("  => 若两者接近，说明'开盘后买入'的收益来自A股自身日内漂移，而非美股信号")

# ---------- 规律3：板块广度共振 ----------
print("\n【规律3】板块共振：缺口高开 + 广度，对日内续涨的影响")
up_gap = df[df["cn_gap"]>0.01].copy()  # 明确高开>1%
hi_breadth = up_gap[up_gap["cn_breadth"]>=0.7]
lo_breadth = up_gap[up_gap["cn_breadth"]<0.5]
print(f"  高开>1% 且 广度>=70%(普涨): n={len(hi_breadth)}  日内均值 {pr(hi_breadth['cn_intraday'].mean())} 胜率 {(hi_breadth['cn_intraday']>0).mean()*100:.0f}%")
print(f"  高开>1% 但 广度<50%(分化):  n={len(lo_breadth)}  日内均值 {pr(lo_breadth['cn_intraday'].mean())} 胜率 {(lo_breadth['cn_intraday']>0).mean()*100:.0f}%")
print("  => 广度高(板块共振) 应显著优于 仅龙头独涨")

# ---------- 规律4：多日漂移（美股强涨夜后 A股 T+1..T+5 累计） ----------
print("\n【规律4】美股强涨夜后 A股全天收益的持续性 (T为触发日=当日)")
sig = df["us_overnight"]
full = df["cn_full"]
trig = df.index[sig>=thi]
horizons = {}
full_arr = full.values; idxmap = {d:i for i,d in enumerate(df.index)}
for h in [0,1,2,3,4]:
    vals=[]
    for d in trig:
        i=idxmap[d]
        if i+h < len(full_arr): vals.append(full_arr[i+h])
    horizons[h]=np.array(vals)
cum=0
for h in [0,1,2,3,4]:
    v=horizons[h]; cum+=np.nanmean(v)
    print(f"  T+{h}: 当日均值 {pr(np.nanmean(v))} 胜率 {(v>0).mean()*100:.0f}%  累计 {pr(cum)}  (n={len(v)})")
print("  => 若 T+1 之后转负/走平，说明利好当日兑现，无多日跟随")

# ---------- 规律5：谁领先谁（同日 vs 滞后 相关） ----------
print("\n【规律5】领先滞后方向检验")
# 重新构造同一日历日下 美股cc 与 A股cc（同日：A股先收盘, 美股后开盘 -> A可能领先美同日）
# us_overnight 已是"对齐到A股日期、且来自更早美股日"的滞后信号 => r=0.250(规律③)
# 这里补一个：A股当日cc 对 美股【同日历日】cc 的相关（检验 A 是否领先美）
tf = TickFlow()
def cc_series(sym):
    d = tf.klines.get(sym, count=600, as_dataframe=True)
    d["trade_date"]=pd.to_datetime(d["trade_date"]); d=d.sort_values("trade_date").set_index("trade_date")
    return d["close"].astype(float).pct_change()
try:
    us_cc = pd.concat([cc_series(s) for s in ["NVDA.US","AVGO.US","COHR.US","LITE.US","FN.US"]],axis=1).mean(axis=1)
    cn_cc = pd.concat([cc_series(s) for s in ["300308.SZ","300502.SZ","300394.SZ"]],axis=1).mean(axis=1)
    m = pd.DataFrame({"us":us_cc,"cn":cn_cc}).dropna()
    r_same = m["us"].corr(m["cn"])                       # 同日历日
    r_us_lead = m["us"].shift(1).corr(m["cn"])           # 美股领先1日 -> A股
    r_cn_lead = m["cn"].shift(1).corr(m["us"])           # A股领先1日 -> 美股
    print(f"  同日历日 corr(美cc, A cc)        = {r_same:+.3f}")
    print(f"  美股领先1日 corr(美cc(t-1),A cc(t))= {r_us_lead:+.3f}  <- 这就是隔夜传导主通道")
    print(f"  A股领先1日 corr(A cc(t-1),美cc(t)) = {r_cn_lead:+.3f}  <- A股是否反向领先美股")
except Exception as e:
    print("  lead-lag err:", repr(e)[:150])

# ---------- 规律6：不对称（美涨vs美跌 对 A股全天的弹性） ----------
print("\n【规律6】传导不对称：分位回归式弹性")
up = df[df["us_overnight"]>0]; dn = df[df["us_overnight"]<0]
bu = np.polyfit(up["us_overnight"],up["cn_full"],1)[0] if len(up)>10 else float('nan')
bd = np.polyfit(dn["us_overnight"],dn["cn_full"],1)[0] if len(dn)>10 else float('nan')
print(f"  美股为正时 A股全天对美股 beta = {bu:.2f}  (n={len(up)})")
print(f"  美股为负时 A股全天对美股 beta = {bd:.2f}  (n={len(dn)})")
print("  => beta(正) > beta(负) 说明'追涨快、杀跌钝'，下行有缓冲")
