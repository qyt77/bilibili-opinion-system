# Public Opinion Analysis

一个用于采集、清洗、分析和可视化舆情评论的 Python 项目骨架。

## 项目结构

- `src/`: 核心代码
- `data/`: 原始评论和清洗后的评论
- `output/`: 图表和分析报告输出
- `config/`: 停用词等配置

## 安装依赖

```bash
pip install -r requirements.txt
```

## 运行

```bash
python src/main.py
```

运行后会读取 `data/raw_comments.csv`，生成清洗数据、情感分析图表和文本报告。
