"""
CPO 中美联动规律实证 — 用 TickFlow 真实日线数据找规律
核心问题：美股 CPO 隔夜涨跌 -> A股次日(缺口/日内/全天)如何反应？是否可交易？

时差对齐逻辑：
  美股 D 日的收盘发生在 北京时间 D+1 凌晨；A股 D+1 日 09:30 开盘才第一次定价。
  因此对 A股交易日 a_date，其"可用美股信息" = 严格早于 a_date 的最近一个美股交易日的 close-to-close 收益。

key 从环境变量 TICKFLOW_API_KEY 读取，不硬编码。
"""
import os, sys, json
import numpy as np
import pandas as pd
from tickflow import TickFlow

COUNT = int(os.environ.get("COUNT", "500"))

US = {
    "NVDA.US": "英伟达(总量锚)", "AVGO.US": "博通", "COHR.US": "Coherent",
    "LITE.US": "Lumentum", "FN.US": "Fabrinet", "CRDO.US": "Credo",
    "MRVL.US": "Marvell", "AAOI.US": "AppliedOpto", "ANET.US": "Arista", "POET.US": "POET",
}
CN = {
    "300308.SZ": "中际旭创", "300502.SZ": "新易盛", "300394.SZ": "天孚通信",
    "002281.SZ": "光迅科技", "300570.SZ": "太辰光", "300548.SZ": "博创科技",
    "688498.SH": "源杰科技", "603083.SH": "剑桥科技", "000988.SZ": "华工科技",
    "300620.SZ": "光库科技",
}

def pull(symbols):
    tf = TickFlow()
    out = {}
    for s in symbols:
        try:
            df = tf.klines.get(s, count=COUNT, as_dataframe=True)
            df = df[["trade_date", "open", "high", "low", "close", "volume", "amount"]].copy()
            df["trade_date"] = pd.to_datetime(df["trade_date"])
            df = df.sort_values("trade_date").drop_duplicates("trade_date").set_index("trade_date")
            for c in ["open", "high", "low", "close", "volume", "amount"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            out[s] = df
        except Exception as e:
            print(f"  WARN pull {s}: {repr(e)[:120]}", file=sys.stderr)
    return out

def basket_daily(data):
    """等权篮子的日度 close-to-close 收益、开盘缺口、日内收益。返回 DataFrame index=date。"""
    cc, gap, intr = {}, {}, {}
    for s, df in data.items():
        prev_close = df["close"].shift(1)
        cc[s] = df["close"] / prev_close - 1          # 全天 收-收
        gap[s] = df["open"] / prev_close - 1          # 隔夜缺口 开/昨收
        intr[s] = df["close"] / df["open"] - 1        # 日内 收/开
    cc = pd.DataFrame(cc); gap = pd.DataFrame(gap); intr = pd.DataFrame(intr)
    res = pd.DataFrame({
        "cc": cc.mean(axis=1, skipna=True),
        "gap": gap.mean(axis=1, skipna=True),
        "intraday": intr.mean(axis=1, skipna=True),
        "breadth_up": (cc > 0).mean(axis=1),   # 上涨家数占比
        "n": cc.notna().sum(axis=1),
    })
    return res.dropna(subset=["cc"])

def align_us_to_cn(us_sig: pd.Series, cn_dates: pd.DatetimeIndex):
    """对每个 A股日期，取严格早于它的最近美股交易日的信号值。"""
    us_sig = us_sig.dropna().sort_index()
    ud = us_sig.index.values
    vals = us_sig.values
    pos = np.searchsorted(ud, cn_dates.values, side="left") - 1  # 严格小于
    out = np.full(len(cn_dates), np.nan)
    mask = pos >= 0
    out[mask] = vals[pos[mask]]
    return pd.Series(out, index=cn_dates)

def stats_block(title, x, y):
    d = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(d) < 20:
        return f"\n[{title}] 样本不足 ({len(d)})"
    r = d["x"].corr(d["y"])
    # 按美股信号方向分组
    up = d[d["x"] > 0]["y"]; dn = d[d["x"] < 0]["y"]
    # 强信号阈值（美股信号 80 分位）
    thi = d["x"].quantile(0.80); tlo = d["x"].quantile(0.20)
    strong_up = d[d["x"] >= thi]["y"]; strong_dn = d[d["x"] <= tlo]["y"]
    def line(name, s):
        if len(s) == 0: return f"    {name}: n=0"
        return (f"    {name}: n={len(s):>3}  均值={s.mean()*100:+.2f}%  "
                f"中位={s.median()*100:+.2f}%  胜率(>0)={ (s>0).mean()*100:.0f}%")
    return (f"\n[{title}]  Pearson r = {r:+.3f}  (n={len(d)})\n"
            f"  美股信号为正日 -> A股 y:\n{line('全部正',up)}\n{line('强正(>=80%)',strong_up)}\n"
            f"  美股信号为负日 -> A股 y:\n{line('全部负',dn)}\n{line('强负(<=20%)',strong_dn)}")

def main():
    if not os.environ.get("TICKFLOW_API_KEY"):
        print("ERROR: set TICKFLOW_API_KEY", file=sys.stderr); sys.exit(1)
    print(f"# 拉取数据 count={COUNT} ...", file=sys.stderr)
    us_data = pull(list(US)); cn_data = pull(list(CN))
    print(f"  US ok={len(us_data)}  CN ok={len(cn_data)}", file=sys.stderr)

    us = basket_daily(us_data)   # 美股篮子日度
    cn = basket_daily(cn_data)   # A股篮子日度

    # 美股"隔夜信号" = 美股篮子当日 close-to-close（在 A股次日开盘前已知）
    us_signal_for_cn = align_us_to_cn(us["cc"], cn.index)

    df = pd.DataFrame({
        "us_overnight": us_signal_for_cn,   # 美股隔夜信号（对齐到A股日期）
        "cn_gap": cn["gap"],                # A股开盘缺口
        "cn_intraday": cn["intraday"],      # A股日内（收/开）
        "cn_full": cn["cc"],                # A股全天（收/昨收）
        "cn_breadth": cn["breadth_up"],     # A股板块广度
    }).dropna(subset=["us_overnight", "cn_full"])

    span = f"{df.index.min().date()} ~ {df.index.max().date()}, N={len(df)}"
    print("="*78)
    print(f"CPO 中美联动实证   样本区间: {span}")
    print(f"美股篮子: {len(us_data)}只   A股篮子: {len(cn_data)}只")
    print("="*78)

    print(stats_block("① 美股隔夜 -> A股开盘缺口 cn_gap", df["us_overnight"], df["cn_gap"]))
    print(stats_block("② 美股隔夜 -> A股日内(收/开) cn_intraday", df["us_overnight"], df["cn_intraday"]))
    print(stats_block("③ 美股隔夜 -> A股全天(收/昨收) cn_full", df["us_overnight"], df["cn_full"]))

    # 核心可交易性：美股大涨夜之后，A股 高开后是续涨还是兑现？
    print("\n" + "="*78)
    print("④ 【可交易性核心】美股强涨夜(信号>=80分位) 之后 A股的处置")
    thi = df["us_overnight"].quantile(0.80)
    big = df[df["us_overnight"] >= thi]
    print(f"  触发日数={len(big)}  阈值(美股篮子隔夜)>= {thi*100:.2f}%")
    if len(big):
        gp = big["cn_gap"]; itd = big["cn_intraday"]; full = big["cn_full"]
        print(f"  A股开盘缺口  : 均值 {gp.mean()*100:+.2f}%  高开比例 {(gp>0).mean()*100:.0f}%")
        print(f"  A股日内收/开 : 均值 {itd.mean()*100:+.2f}%  日内翻红比例 {(itd>0).mean()*100:.0f}%")
        print(f"  A股全天收益  : 均值 {full.mean()*100:+.2f}%  全天红盘比例 {(full>0).mean()*100:.0f}%")
        # 高开低走（兑现）比例：缺口为正但日内为负
        fade = big[(big["cn_gap"] > 0) & (big["cn_intraday"] < 0)]
        cont = big[(big["cn_gap"] > 0) & (big["cn_intraday"] > 0)]
        print(f"  高开后【兑现/低走】(gap>0 & 日内<0) 占比: {len(fade)/max(len(big),1)*100:.0f}%")
        print(f"  高开后【续涨/高走】(gap>0 & 日内>0) 占比: {len(cont)/max(len(big),1)*100:.0f}%")
        # 一个朴素策略：美股强涨夜，次日09:30开盘价买入，收盘卖出 -> 收益分布 = cn_intraday
        s = itd.dropna()
        print(f"  >> 朴素策略[强涨夜→开盘买入→收盘卖出] 期望日内收益 {s.mean()*100:+.2f}% "
              f"  胜率 {(s>0).mean()*100:.0f}%  最好 {s.max()*100:+.2f}% / 最差 {s.min()*100:+.2f}%")

    # 对比：美股强跌夜
    print("\n⑤ 对照：美股强跌夜(信号<=20分位) 之后 A股")
    tlo = df["us_overnight"].quantile(0.20)
    bad = df[df["us_overnight"] <= tlo]
    if len(bad):
        print(f"  触发日数={len(bad)}  阈值<= {tlo*100:.2f}%")
        print(f"  A股缺口均值 {bad['cn_gap'].mean()*100:+.2f}%  日内均值 {bad['cn_intraday'].mean()*100:+.2f}%  全天均值 {bad['cn_full'].mean()*100:+.2f}%")

    # 缺口里"截留"了多少信息：gap 占 full 的比例
    print("\n⑥ 缺口截留效应：A股全天收益中，开盘缺口贡献占比")
    pos_full = df[df["cn_full"] > 0]
    if len(pos_full):
        share = (pos_full["cn_gap"] / pos_full["cn_full"]).clip(-3, 3)
        print(f"  红盘日中 缺口/全天 中位数 = {share.median()*100:.0f}%  (越高=越多收益在你能买之前已发生)")

    # 个股层面：哪些A股对美股信号最敏感（次日全天 beta）
    print("\n⑦ 个股敏感度：各A股【次日全天收益】对【美股隔夜信号】的回归斜率(beta)与相关")
    rows = []
    for s, d in cn_data.items():
        prev = d["close"].shift(1)
        full = (d["close"]/prev - 1)
        xs = align_us_to_cn(us["cc"], full.index)
        dd = pd.DataFrame({"x": xs, "y": full}).dropna()
        if len(dd) >= 30:
            beta = np.polyfit(dd["x"], dd["y"], 1)[0]
            rows.append((CN[s], s, dd["x"].corr(dd["y"]), beta, len(dd)))
    rows.sort(key=lambda r: -r[2])
    print(f"  {'名称':<8}{'代码':<12}{'相关r':>8}{'beta':>8}{'n':>6}")
    for nm, s, r, b, n in rows:
        print(f"  {nm:<8}{s:<12}{r:>8.3f}{b:>8.2f}{n:>6}")

    # 保存对齐后的明细，供进一步看
    df.to_csv("CPO板块/quant/aligned_daily.csv")
    print("\n[saved] CPO板块/quant/aligned_daily.csv")

if __name__ == "__main__":
    main()
