import os
import pandas as pd


class ReportGenerator:
    """
    舆情分析报告生成器：正式报告版

    特点：
    1. 不再像调试日志一样堆 SnowNLP、规则、AI字段
    2. 报告更像课程作业/系统输出报告
    3. 保留总体结论、图表说明、高影响力评论、风险建议
    4. 代表性评论只展示必要信息：评论、情感类型、点赞数、影响力分数
    """

    def __init__(self):
        self.sentiment_path = os.path.join("data", "sentiment_comments.csv")
        self.output_path = os.path.join("output", "final_analysis_report.txt")

    def load_data(self) -> pd.DataFrame:
        if not os.path.exists(self.sentiment_path):
            raise FileNotFoundError(f"未找到情感分析结果文件：{self.sentiment_path}")

        df = pd.read_csv(self.sentiment_path, encoding="utf-8-sig")

        required_columns = [
            "clean_text",
            "sentiment_type",
            "sentiment_score",
            "like_count",
            "influence_score"
        ]

        for column in required_columns:
            if column not in df.columns:
                raise ValueError(f"情感分析结果缺少必要字段：{column}")

        df["sentiment_score"] = pd.to_numeric(
            df["sentiment_score"],
            errors="coerce"
        ).fillna(0.5)

        df["like_count"] = pd.to_numeric(
            df["like_count"],
            errors="coerce"
        ).fillna(0)

        df["influence_score"] = pd.to_numeric(
            df["influence_score"],
            errors="coerce"
        ).fillna(0)

        if "comment_time" in df.columns:
            df["comment_time"] = pd.to_datetime(
                df["comment_time"],
                errors="coerce"
            )

        if "ai_reviewed" not in df.columns:
            df["ai_reviewed"] = False

        return df

    def get_basic_metrics(self, df: pd.DataFrame) -> dict:
        total = len(df)

        sentiment_counts = (
            df["sentiment_type"]
            .value_counts()
            .reindex(["正向", "中立", "负向"], fill_value=0)
        )

        positive_count = int(sentiment_counts["正向"])
        neutral_count = int(sentiment_counts["中立"])
        negative_count = int(sentiment_counts["负向"])

        positive_ratio = round(positive_count / total * 100, 2) if total else 0
        neutral_ratio = round(neutral_count / total * 100, 2) if total else 0
        negative_ratio = round(negative_count / total * 100, 2) if total else 0

        avg_score = round(float(df["sentiment_score"].mean()), 4) if total else 0
        median_score = round(float(df["sentiment_score"].median()), 4) if total else 0

        strong_positive_count = int((df["sentiment_score"] >= 0.8).sum())
        strong_negative_count = int((df["sentiment_score"] <= 0.2).sum())

        strong_positive_ratio = round(strong_positive_count / total * 100, 2) if total else 0
        strong_negative_ratio = round(strong_negative_count / total * 100, 2) if total else 0

        dispute_index = round(
            1 - abs(positive_ratio - negative_ratio) / 100,
            4
        )

        avg_like = round(float(df["like_count"].mean()), 2) if total else 0
        max_like = int(df["like_count"].max()) if total else 0

        ai_review_count = int(
            df["ai_reviewed"].astype(str).str.lower().eq("true").sum()
        )

        return {
            "total": total,
            "positive_count": positive_count,
            "neutral_count": neutral_count,
            "negative_count": negative_count,
            "positive_ratio": positive_ratio,
            "neutral_ratio": neutral_ratio,
            "negative_ratio": negative_ratio,
            "avg_score": avg_score,
            "median_score": median_score,
            "strong_positive_count": strong_positive_count,
            "strong_negative_count": strong_negative_count,
            "strong_positive_ratio": strong_positive_ratio,
            "strong_negative_ratio": strong_negative_ratio,
            "dispute_index": dispute_index,
            "avg_like": avg_like,
            "max_like": max_like,
            "ai_review_count": ai_review_count
        }

    def get_overall_judgement(self, metrics: dict) -> str:
        positive_ratio = metrics["positive_ratio"]
        negative_ratio = metrics["negative_ratio"]
        neutral_ratio = metrics["neutral_ratio"]
        avg_score = metrics["avg_score"]
        dispute_index = metrics["dispute_index"]

        if positive_ratio >= 60:
            tone = (
                "本次评论区整体以正向情绪为主。多数评论表现出认可、支持、共鸣或积极表达，"
                "说明视频内容或相关讨论对象在样本评论中获得了较多正面反馈。"
            )
        elif negative_ratio >= 45:
            tone = (
                "本次评论区负向情绪较为突出。评论中存在较明显的质疑、不满、焦虑或批评表达，"
                "说明该视频评论区存在一定舆情风险，需要重点关注负向评论的具体内容。"
            )
        elif positive_ratio > negative_ratio:
            tone = (
                "本次评论区整体略偏正向，但负向评论仍占有一定比例。"
                "这说明评论区主流态度相对积极，但仍存在不同意见和一定争议。"
            )
        else:
            tone = (
                "本次评论区正负向态度较为接近，整体舆情呈现分化特征。"
                "不同用户之间存在明显观点差异，需要结合高影响力评论进一步分析。"
            )

        if dispute_index >= 0.85:
            dispute = (
                "从争议度来看，正负向评论比例较接近，说明评论区存在较明显的讨论分歧。"
            )
        elif dispute_index >= 0.65:
            dispute = (
                "从争议度来看，评论区存在一定争议，但仍能观察到相对主要的情绪方向。"
            )
        else:
            dispute = (
                "从争议度来看，评论区观点较为集中，主流情绪方向较明确。"
            )

        score_desc = (
            f"平均情感得分为 {avg_score}，中立评论占比为 {neutral_ratio}%。"
            "平均情感得分越接近 1，说明整体情绪越偏正向；越接近 0，则说明整体情绪越偏负向。"
        )

        return tone + dispute + score_desc

    def get_sentiment_structure_analysis(self, metrics: dict) -> str:
        return (
            f"在 {metrics['total']} 条有效评论中，正向评论 {metrics['positive_count']} 条，"
            f"占比 {metrics['positive_ratio']}%；中立评论 {metrics['neutral_count']} 条，"
            f"占比 {metrics['neutral_ratio']}%；负向评论 {metrics['negative_count']} 条，"
            f"占比 {metrics['negative_ratio']}%。"
            f"其中，强正向评论 {metrics['strong_positive_count']} 条，"
            f"占比 {metrics['strong_positive_ratio']}%；强负向评论 {metrics['strong_negative_count']} 条，"
            f"占比 {metrics['strong_negative_ratio']}%。"
            "情感结构可以反映评论区的总体态度分布，也能帮助判断评论区是否存在明显负面集中或态度分化。"
        )

    def build_trend_data(self, df: pd.DataFrame):
        if "comment_time" not in df.columns:
            return pd.DataFrame(), ""

        trend_df = df.dropna(subset=["comment_time"]).copy()

        if trend_df.empty:
            return pd.DataFrame(), ""

        min_time = trend_df["comment_time"].min()
        max_time = trend_df["comment_time"].max()
        time_span = max_time - min_time

        if time_span <= pd.Timedelta(days=2):
            trend_df["time_group"] = trend_df["comment_time"].dt.floor("h")
            group_type = "按小时"
        else:
            trend_df["time_group"] = trend_df["comment_time"].dt.date
            group_type = "按日期"

        trend = (
            trend_df.groupby("time_group")
            .agg(
                avg_score=("sentiment_score", "mean"),
                comment_count=("sentiment_score", "count")
            )
            .reset_index()
        )

        trend["avg_score"] = trend["avg_score"].round(4)

        return trend, group_type

    def get_trend_analysis(self, trend: pd.DataFrame, group_type: str) -> str:
        if trend.empty:
            return (
                "由于评论数据中缺少有效时间字段，本次报告未进行情绪时间趋势分析。"
            )

        if len(trend) <= 1:
            return (
                "本次评论样本的时间跨度较短，分组后时间节点较少，因此情绪时间趋势不明显。"
                "这通常说明采集到的评论集中在较短时间范围内，不代表程序异常。"
            )

        first_score = float(trend["avg_score"].iloc[0])
        last_score = float(trend["avg_score"].iloc[-1])
        max_score = float(trend["avg_score"].max())
        min_score = float(trend["avg_score"].min())

        change = round(last_score - first_score, 4)
        fluctuation = round(max_score - min_score, 4)

        if change > 0.03:
            direction = (
                f"从时间趋势看，评论平均情感得分由 {first_score:.4f} 上升至 {last_score:.4f}，"
                "整体情绪呈上升趋势。"
            )
        elif change < -0.03:
            direction = (
                f"从时间趋势看，评论平均情感得分由 {first_score:.4f} 下降至 {last_score:.4f}，"
                "整体情绪呈下降趋势。"
            )
        else:
            direction = (
                f"从时间趋势看，评论平均情感得分由 {first_score:.4f} 变化至 {last_score:.4f}，"
                "整体情绪变化较小。"
            )

        if fluctuation >= 0.35:
            fluctuation_desc = (
                f"最高与最低平均得分差值为 {fluctuation:.4f}，说明不同时间段之间情绪波动较明显。"
            )
        elif fluctuation >= 0.15:
            fluctuation_desc = (
                f"最高与最低平均得分差值为 {fluctuation:.4f}，说明评论情绪存在一定波动。"
            )
        else:
            fluctuation_desc = (
                f"最高与最低平均得分差值为 {fluctuation:.4f}，说明评论情绪整体较为平稳。"
            )

        return f"情绪时间趋势图采用{group_type}统计。{direction}{fluctuation_desc}"

    def get_high_influence_comments(self, df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
        return (
            df.sort_values(by="influence_score", ascending=False)
            .head(top_n)
            .copy()
        )

    def get_high_influence_analysis(self, df: pd.DataFrame) -> str:
        top_df = self.get_high_influence_comments(df, top_n=10)

        if top_df.empty:
            return "当前样本中没有可用于分析的高影响力评论。"

        type_counts = (
            top_df["sentiment_type"]
            .value_counts()
            .reindex(["正向", "中立", "负向"], fill_value=0)
        )

        positive_count = int(type_counts["正向"])
        neutral_count = int(type_counts["中立"])
        negative_count = int(type_counts["负向"])

        avg_like = round(float(top_df["like_count"].mean()), 2)
        max_like = int(top_df["like_count"].max())

        if negative_count > positive_count:
            direction = (
                "高影响力评论中负向评论数量相对较多，说明部分获得较多点赞的评论带有质疑、批评或不满情绪。"
                "这类评论可能对评论区氛围产生较大影响，应作为舆情风险观察重点。"
            )
        elif positive_count > negative_count:
            direction = (
                "高影响力评论中正向评论数量相对较多，说明点赞较高的评论整体偏积极，"
                "对评论区氛围可能具有正向带动作用。"
            )
        else:
            direction = (
                "高影响力评论中正向与负向评论数量较为接近，说明高赞评论内部也存在态度差异。"
            )

        return (
            f"高影响力评论按照“点赞数 × 情绪极端程度”计算。"
            f"前 10 条高影响力评论中，正向 {positive_count} 条，"
            f"中立 {neutral_count} 条，负向 {negative_count} 条；"
            f"平均点赞数为 {avg_like}，最高点赞数为 {max_like}。"
            f"{direction}"
        )

    def get_risk_suggestion(self, metrics: dict, df: pd.DataFrame) -> str:
        negative_ratio = metrics["negative_ratio"]
        dispute_index = metrics["dispute_index"]

        top10 = self.get_high_influence_comments(df, top_n=10)

        if top10.empty:
            high_negative_count = 0
        else:
            high_negative_count = int((top10["sentiment_type"] == "负向").sum())

        if negative_ratio >= 35 or dispute_index >= 0.8 or high_negative_count >= 3:
            return (
                "综合负向评论占比、争议度指数和高影响力评论情况来看，本次评论区存在一定舆情风险。"
                "建议重点关注高赞负向评论和高影响力负向评论，进一步分析用户不满或质疑的具体来源。"
                "如果该系统用于实际舆情监测，应持续跟踪后续评论变化，判断负面情绪是否进一步扩散。"
            )

        return (
            "综合负向评论占比、争议度指数和高影响力评论情况来看，本次评论区整体风险相对可控。"
            "后续仍可关注最新评论变化，尤其是高点赞评论是否出现明显负面转向。"
        )

    def get_chart_description(self) -> str:
        return (
            "本次系统生成的主分析图包括：情感类别分布图、情感得分分布图、情绪时间趋势图和高影响力评论排行图。"
            "其中，情感类别分布图用于观察正向、中立、负向评论数量结构；情感得分分布图用于观察评论情绪集中区间；"
            "情绪时间趋势图用于观察评论情绪随时间变化情况；高影响力评论排行图用于发现可能影响评论区氛围的重点评论。"
            "此外，系统还生成情感环形图、正向评论词云和负向评论词云作为补充展示。"
            "词云图容易受到分词效果、口语词和背景词影响，因此本报告将其作为展示材料，不作为核心结论依据。"
        )

    def format_comment_section(self, df: pd.DataFrame) -> str:
        if df.empty:
            return "暂无高影响力评论。\n"

        lines = []

        for index, (_, row) in enumerate(df.iterrows(), start=1):
            comment = str(row.get("clean_text", "")).strip()
            sentiment_type = row.get("sentiment_type", "未知")
            like_count = int(row.get("like_count", 0))
            influence_score = row.get("influence_score", 0)

            lines.append(
                f"{index}. {comment}\n"
                f"   情感类型：{sentiment_type}；点赞数：{like_count}；影响力分数：{influence_score}\n"
            )

        return "\n".join(lines)

    def generate_report_text(self, df: pd.DataFrame) -> str:
        metrics = self.get_basic_metrics(df)
        trend, group_type = self.build_trend_data(df)
        high_influence_comments = self.get_high_influence_comments(df, top_n=5)

        overall_judgement = self.get_overall_judgement(metrics)
        sentiment_structure = self.get_sentiment_structure_analysis(metrics)
        trend_analysis = self.get_trend_analysis(trend, group_type)
        influence_analysis = self.get_high_influence_analysis(df)
        risk_suggestion = self.get_risk_suggestion(metrics, df)
        chart_description = self.get_chart_description()
        comment_section = self.format_comment_section(high_influence_comments)

        lines = []

        lines.append("========== B站评论舆情分析报告 ==========\n\n")

        lines.append("一、报告概述\n")
        lines.append(
            "本报告基于采集并清洗后的 B站 评论数据生成，系统完成了评论采集、文本清洗、情感分析、"
            "关键评论 AI 复核、可视化图表生成和舆情报告输出等流程。"
            "报告主要用于观察当前样本评论区的总体情绪倾向、争议程度以及可能影响舆论氛围的重点评论。\n\n"
        )

        lines.append("二、样本与指标统计\n")
        lines.append(f"有效评论数量：{metrics['total']} 条\n")
        lines.append(f"平均情感得分：{metrics['avg_score']}\n")
        lines.append(f"情感得分中位数：{metrics['median_score']}\n")
        lines.append(f"正向评论：{metrics['positive_count']} 条，占比 {metrics['positive_ratio']}%\n")
        lines.append(f"中立评论：{metrics['neutral_count']} 条，占比 {metrics['neutral_ratio']}%\n")
        lines.append(f"负向评论：{metrics['negative_count']} 条，占比 {metrics['negative_ratio']}%\n")
        lines.append(f"强正向评论：{metrics['strong_positive_count']} 条，占比 {metrics['strong_positive_ratio']}%\n")
        lines.append(f"强负向评论：{metrics['strong_negative_count']} 条，占比 {metrics['strong_negative_ratio']}%\n")
        lines.append(f"争议度指数：{metrics['dispute_index']}\n")
        lines.append(f"平均点赞数：{metrics['avg_like']}\n")
        lines.append(f"最高点赞数：{metrics['max_like']}\n")
        lines.append(f"AI复核评论数量：{metrics['ai_review_count']} 条\n\n")

        lines.append("三、总体舆情判断\n")
        lines.append(overall_judgement + "\n\n")

        lines.append("四、情感结构分析\n")
        lines.append(sentiment_structure + "\n\n")

        lines.append("五、情绪时间趋势分析\n")
        lines.append(trend_analysis + "\n\n")

        lines.append("六、高影响力评论分析\n")
        lines.append(influence_analysis + "\n\n")

        lines.append("七、高影响力评论示例\n")
        lines.append(comment_section + "\n")

        lines.append("八、图表说明\n")
        lines.append(chart_description + "\n\n")

        lines.append("九、舆情风险与建议\n")
        lines.append(risk_suggestion + "\n\n")

        lines.append("十、方法说明与局限\n")
        lines.append(
            "本项目使用 SnowNLP 作为基础情感分析工具，并结合规则修正和 Gemini AI 关键评论复核提升复杂评论的判断效果。"
            "由于网络评论中存在反讽、调侃、表情包、语境省略等情况，单条评论的情感识别仍可能存在误差。"
            "因此，本报告更关注整体情感分布、时间趋势、高影响力评论和争议度变化，而不将每一条评论的分类视为绝对准确。"
        )

        return "".join(lines)

    def save_report(self, report_text: str):
        output_dir = os.path.dirname(self.output_path)

        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(self.output_path, "w", encoding="utf-8") as file:
            file.write(report_text)

        print(f"最终舆情分析报告已生成：{self.output_path}")

    def run(self):
        df = self.load_data()
        report_text = self.generate_report_text(df)
        self.save_report(report_text)


if __name__ == "__main__":
    generator = ReportGenerator()
    generator.run()