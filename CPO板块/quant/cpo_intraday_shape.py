"""第三轮：日内走势形态。用日线OHLC重构日内路径，在'美股强涨夜'事件上统计形态分布。
路径代理指标：
  gap        = open/prevclose-1           隔夜缺口
  runup      = high/open-1                开盘后最大冲高
  drawdown   = open/low-1                 开盘后最大回撤(正数=向下)
  close_loc  = (close-low)/(high-low)     收盘在全天振幅中的位置(1=收最高,0=收最低)
  intraday   = close/open-1               日内收/开
形态分类(基于龙头中际旭创为主+篮子)：
  A 高开高走 : gap>0 & intraday>0 & close_loc>0.6
  B 高开低走 : gap>0 & intraday<0           (兑现)
  C 低开高走 : gap<0 & intraday>0           (修复)
  D 平开震荡 : |gap|<=0.3% & |intraday|<=0.5%
  E 冲高回落 : runup>1.5% & close_loc<0.3   (盘中冲高后大幅回落)
"""
import os
import numpy as np
import pandas as pd
from tickflow import TickFlow

tf = TickFlow()
CN = {"300308.SZ":"中际旭创","300502.SZ":"新易盛","300394.SZ":"天孚通信",
      "300548.SZ":"博创科技","300570.SZ":"太辰光"}
US = ["NVDA.US","AVGO.US","COHR.US","LITE.US","FN.US"]

def daily(sym):
    d = tf.klines.get(sym, count=600, as_dataframe=True)
    d["trade_date"]=pd.to_datetime(d["trade_date"])
    d=d.sort_values("trade_date").drop_duplicates("trade_date").set_index("trade_date")
    for c in ["open","high","low","close"]: d[c]=pd.to_numeric(d[c],errors="coerce")
    return d

# 美股篮子隔夜信号
us_cc = pd.concat([daily(s)["close"].pct_change() for s in US],axis=1).mean(axis=1).dropna()

def align(sig, dates):
    sig=sig.sort_index(); ud=sig.index.values; v=sig.values
    pos=np.searchsorted(ud,dates.values,side="left")-1
    out=np.full(len(dates),np.nan); m=pos>=0; out[m]=v[pos[m]]
    return pd.Series(out,index=dates)

# 用龙头中际旭创做形态主体（也代表篮子方向）
lead = daily("300308.SZ")
pc = lead["close"].shift(1)
path = pd.DataFrame({
    "gap": lead["open"]/pc-1,
    "runup": lead["high"]/lead["open"]-1,
    "drawdown": lead["open"]/lead["low"]-1,
    "close_loc": (lead["close"]-lead["low"])/(lead["high"]-lead["low"]).replace(0,np.nan),
    "intraday": lead["close"]/lead["open"]-1,
}).dropna()
path["us"] = align(us_cc, path.index)
path=path.dropna(subset=["us"])

def classify(r):
    if abs(r.gap)<=0.003 and abs(r.intraday)<=0.005: return "D平开震荡"
    if r.runup>0.015 and r.close_loc<0.3: return "E冲高回落"
    if r.gap>0 and r.intraday>0 and r.close_loc>0.6: return "A高开高走"
    if r.gap>0 and r.intraday<0: return "B高开低走/兑现"
    if r.gap<0 and r.intraday>0: return "C低开高走/修复"
    if r.gap>0 and r.intraday>=0: return "A-高开温和走高"
    return "其他/低开低走"
path["pattern"]=path.apply(classify,axis=1)

thi=path["us"].quantile(0.80)
print("="*72)
print(f"日内形态分布(龙头 300308)  N={len(path)}  美股强涨阈值>= {thi*100:.2f}%")
print("="*72)

def dist(sub,title):
    print(f"\n【{title}】 n={len(sub)}")
    vc=sub["pattern"].value_counts(normalize=True)*100
    for k,v in vc.items(): print(f"   {k:<16}{v:5.0f}%")
    print(f"   平均: 缺口{sub['gap'].mean()*100:+.2f}%  冲高{sub['runup'].mean()*100:+.2f}%  "
          f"回撤-{sub['drawdown'].mean()*100:.2f}%  收盘位置{sub['close_loc'].mean():.2f}  日内{sub['intraday'].mean()*100:+.2f}%")

dist(path, "全样本")
dist(path[path["us"]>=thi], "美股强涨夜后")
dist(path[path["us"]<=path["us"].quantile(0.20)], "美股强跌夜后")

# 强涨夜后，盈利兑现的择时含义：开盘后冲高幅度 vs 回撤
big=path[path["us"]>=thi]
print("\n【强涨夜后 日内择时含义】")
print(f"  开盘后平均还能冲高 {big['runup'].mean()*100:+.2f}% (中位 {big['runup'].median()*100:+.2f}%)")
print(f"  开盘后平均最大回撤 -{big['drawdown'].mean()*100:.2f}% (中位 -{big['drawdown'].median()*100:.2f}%)")
print(f"  收盘在全天高位(loc>0.6)比例 {(big['close_loc']>0.6).mean()*100:.0f}%  低位(loc<0.3)比例 {(big['close_loc']<0.3).mean()*100:.0f}%")

# 抓最近一个完整交易日的分钟线做真实例证
print("\n【最近交易日 300308 分钟走势例证(每30分钟采样)】")
try:
    m=tf.klines.intraday("300308.SZ",period="1m",count=240,as_dataframe=True)
    m["trade_time"]=pd.to_datetime(m["trade_time"]); m=m.sort_values("trade_time")
    op=m["open"].iloc[0]
    s=m.set_index("trade_time")["close"].astype(float)
    for t in s.iloc[::30].index:
        print(f"   {t.strftime('%H:%M')}  {s[t]:.2f}  ({(s[t]/op-1)*100:+.2f}% vs 开盘)")
except Exception as e:
    print("  intraday err:", repr(e)[:120])
