# CPO 中美联动 · 量化脚本

真实数据来自 TickFlow API。**API key 不硬编码**,从环境变量读取。

## 运行

```bash
pip install tickflow pandas numpy scipy matplotlib
export TICKFLOW_API_KEY="你的key"     # tk_ 开头；请勿提交进仓库
export COUNT=600                       # 拉取日线根数(可选)

python CPO板块/quant/cpo_leadlag.py        # 1) 主联动统计 -> 生成 aligned_daily.csv
python CPO板块/quant/cpo_patterns.py       # 2) 缺口分桶/持续性/领先滞后/不对称
python CPO板块/quant/cpo_intraday_shape.py # 3) 日内形态分布 + 分钟线例证
python CPO板块/quant/make_charts.py        # 4) 图表 -> CPO板块/charts/
```

## 脚本职责

| 文件                    | 作用                                                                            |
| ----------------------- | ------------------------------------------------------------------------------- |
| `cpo_leadlag.py`        | 拉两地各10只篮子,时差对齐,算 美股隔夜→A股 缺口/日内/全天 的相关、胜率、个股beta |
| `cpo_patterns.py`       | 缺口大小分桶、(修正的)广度、多日持续性、领先滞后方向、上下行不对称              |
| `cpo_intraday_shape.py` | 用日线OHLC重构日内路径,统计5类形态分布;抓最近一日分钟线                         |
| `make_charts.py`        | 收益分解图、散点图、逐年耦合图                                                  |

## TickFlow 接口备忘

- base URL: `https://api.tickflow.org/v1`,鉴权头 `x-api-key`
- 代码格式:美股 `AAPL.US`,沪市 `600000.SH`,深市 `000001.SZ`
- SDK:`from tickflow import TickFlow; tf=TickFlow(); tf.klines.get("300308.SZ", count=600, as_dataframe=True)`

## 安全

对话中曾明文出现过 key,**请在 TickFlow 控制台轮换**。本目录任何文件均未写入 key。
