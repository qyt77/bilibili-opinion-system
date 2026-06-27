import os

import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud


class CommentVisualizer:
    """
    评论可视化类：稳定分析版 + 补充展示图。

    主分析图：
    1. sentiment_type_bar.png         情感类别分布柱状图
    2. sentiment_score_hist.png       情感得分频数分布图
    3. sentiment_trend.png            平均情感得分时间趋势图
    4. influence_comments_bar.png     高影响力评论排行图

    补充展示图：
    5. sentiment_pie.png              情感分布环形图
    6. positive_wordcloud.png         正向评论词云
    7. negative_wordcloud.png         负向评论词云

    说明：
    关键词图和词云不作为核心分析依据。
    词云仅作为展示补充，核心分析仍然依赖情感分布、情感得分、时间趋势和高影响力评论。
    """

    def __init__(self):
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
        plt.rcParams["axes.unicode_minus"] = False
        plt.rcParams["figure.facecolor"] = "white"
        plt.rcParams["axes.facecolor"] = "white"
        plt.rcParams["savefig.facecolor"] = "white"

        self.font_path = self.get_chinese_font_path()

        self.colors = {
            "main": "#355C7D",
            "positive": "#4C78A8",
            "neutral": "#B8B8B8",
            "negative": "#C44E52",
            "accent": "#DD8452",
            "dark": "#222222",
            "gray": "#666666",
            "grid": "#D9D9D9"
        }

    def get_chinese_font_path(self):
        """
        获取中文字体路径，兼容 Windows 本地环境和 Streamlit Cloud/Linux 环境。
        """
        possible_fonts = [
            r"C:\Windows\Fonts\msyh.ttc",
            r"C:\Windows\Fonts\simhei.ttf",
            r"C:\Windows\Fonts\simsun.ttc",

            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        ]

        for font in possible_fonts:
            if os.path.exists(font):
                return font

        print("警告：未找到中文字体，词云中文可能显示为方框。")
        return None

    def load_sentiment_data(self, sentiment_path: str) -> pd.DataFrame:
        """
        读取情感分析后的评论数据。
        """
        if not os.path.exists(sentiment_path):
            raise FileNotFoundError(f"未找到情感分析文件：{sentiment_path}")

        df = pd.read_csv(sentiment_path, encoding="utf-8-sig")

        required_columns = [
            "clean_text",
            "sentiment_score",
            "sentiment_type"
        ]

        for column in required_columns:
            if column not in df.columns:
                raise ValueError(f"数据中缺少字段：{column}")

        df["sentiment_score"] = pd.to_numeric(
            df["sentiment_score"],
            errors="coerce"
        ).fillna(0.5)

        if "like_count" in df.columns:
            df["like_count"] = pd.to_numeric(
                df["like_count"],
                errors="coerce"
            ).fillna(0)
        else:
            df["like_count"] = 0

        if "influence_score" in df.columns:
            df["influence_score"] = pd.to_numeric(
                df["influence_score"],
                errors="coerce"
            ).fillna(0)
        else:
            df["influence_score"] = (
                df["like_count"] * abs(df["sentiment_score"] - 0.5)
            ).round(4)

        print(f"成功读取情感分析数据：{sentiment_path}")
        print(f"评论数量：{len(df)} 条")

        return df

    def ensure_output_dir(self, output_dir: str):
        """
        确保输出文件夹存在。
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def save_figure(self, save_path: str):
        """
        保存高清图片。
        """
        try:
            plt.tight_layout()
            plt.savefig(save_path, dpi=320, bbox_inches="tight")
            print(f"图表已保存：{save_path}")
        except PermissionError:
            print(f"保存失败：{save_path} 可能正在被打开，请关闭后重试。")
        finally:
            plt.close()

    def style_axes(self, ax, grid_axis="y"):
        """
        统一学术风坐标轴样式。
        """
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.spines["left"].set_color("#B0B0B0")
        ax.spines["bottom"].set_color("#B0B0B0")

        ax.tick_params(axis="both", labelsize=10, colors="#333333")

        if grid_axis:
            ax.grid(
                axis=grid_axis,
                linestyle="--",
                alpha=0.25,
                color=self.colors["grid"]
            )
            ax.set_axisbelow(True)

    def add_title(self, ax, title: str, subtitle: str = ""):
        """
        添加正式图表标题。
        """
        if subtitle:
            ax.set_title(
                f"{title}\n{subtitle}",
                fontsize=15,
                fontweight="bold",
                color=self.colors["dark"],
                pad=16
            )
        else:
            ax.set_title(
                title,
                fontsize=15,
                fontweight="bold",
                color=self.colors["dark"],
                pad=16
            )

    def draw_sentiment_type_bar(self, df: pd.DataFrame, save_path: str):
        """
        绘制情感类别分布柱状图。
        """
        counts = (
            df["sentiment_type"]
            .value_counts()
            .reindex(["正向", "中立", "负向"], fill_value=0)
        )

        total = counts.sum()
        labels = counts.index.tolist()
        values = counts.values.tolist()
        ratios = [value / total * 100 for value in values]

        bar_colors = [
            self.colors["positive"],
            self.colors["neutral"],
            self.colors["negative"]
        ]

        fig, ax = plt.subplots(figsize=(7.5, 5.4))

        bars = ax.bar(
            labels,
            values,
            color=bar_colors,
            width=0.52,
            alpha=0.92
        )

        for bar, value, ratio in zip(bars, values, ratios):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(values) * 0.025,
                f"{value}条\n{ratio:.1f}%",
                ha="center",
                va="bottom",
                fontsize=10,
                color=self.colors["dark"]
            )

        ax.set_ylabel("评论数量", fontsize=11)
        ax.set_ylim(0, max(values) * 1.22 if max(values) > 0 else 1)

        self.style_axes(ax, grid_axis="y")
        self.add_title(
            ax,
            "图1  评论情感类别分布",
            "统计正向、中立和负向评论数量及占比"
        )

        self.save_figure(save_path)

    def draw_sentiment_pie(self, df: pd.DataFrame, save_path: str):
        """
        绘制情感分布环形图，作为补充展示图。
        """
        counts = (
            df["sentiment_type"]
            .value_counts()
            .reindex(["正向", "中立", "负向"], fill_value=0)
        )

        total = counts.sum()
        labels = counts.index.tolist()
        values = counts.values.tolist()

        pie_colors = [
            self.colors["positive"],
            self.colors["neutral"],
            self.colors["negative"]
        ]

        fig, ax = plt.subplots(figsize=(7.6, 6.2))

        wedges, texts, autotexts = ax.pie(
            values,
            labels=None,
            autopct=lambda pct: f"{pct:.1f}%" if pct > 0 else "",
            startangle=90,
            counterclock=False,
            colors=pie_colors,
            pctdistance=0.78,
            wedgeprops={
                "width": 0.38,
                "edgecolor": "white",
                "linewidth": 2
            }
        )

        for autotext in autotexts:
            autotext.set_fontsize(11)
            autotext.set_color("white")
            autotext.set_fontweight("bold")

        centre_circle = plt.Circle((0, 0), 0.56, color="white")
        ax.add_artist(centre_circle)

        ax.text(
            0,
            0.06,
            f"{total}",
            ha="center",
            va="center",
            fontsize=24,
            fontweight="bold",
            color=self.colors["dark"]
        )

        ax.text(
            0,
            -0.12,
            "有效评论",
            ha="center",
            va="center",
            fontsize=11,
            color=self.colors["gray"]
        )

        legend_labels = [
            f"{label}：{count} 条（{count / total * 100:.1f}%）"
            for label, count in zip(labels, values)
        ]

        ax.legend(
            wedges,
            legend_labels,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=False,
            fontsize=11
        )

        ax.set_title(
            "评论情感分布环形图",
            fontsize=16,
            fontweight="bold",
            color=self.colors["dark"],
            pad=18
        )

        ax.set_aspect("equal")
        self.save_figure(save_path)

    def draw_score_hist(self, df: pd.DataFrame, save_path: str):
        """
        绘制情感得分频数分布图。
        """
        scores = df["sentiment_score"]
        avg_score = scores.mean()

        fig, ax = plt.subplots(figsize=(8.4, 5.6))

        ax.hist(
            scores,
            bins=20,
            color=self.colors["main"],
            alpha=0.88,
            edgecolor="white",
            linewidth=1.0
        )

        ax.axvline(
            avg_score,
            color=self.colors["accent"],
            linestyle="--",
            linewidth=1.8,
            label=f"平均得分 = {avg_score:.2f}"
        )

        ax.axvline(
            0.4,
            color=self.colors["negative"],
            linestyle=":",
            linewidth=1.4,
            alpha=0.8,
            label="负向阈值 = 0.4"
        )

        ax.axvline(
            0.6,
            color=self.colors["positive"],
            linestyle=":",
            linewidth=1.4,
            alpha=0.8,
            label="正向阈值 = 0.6"
        )

        ax.set_xlabel("情感得分", fontsize=11)
        ax.set_ylabel("评论数量", fontsize=11)
        ax.set_xlim(0, 1)

        ax.legend(frameon=False, fontsize=9, loc="upper left")

        self.style_axes(ax, grid_axis="y")
        self.add_title(
            ax,
            "图2  评论情感得分频数分布",
            "得分越接近 1 表示越正向，越接近 0 表示越负向"
        )

        self.save_figure(save_path)

    def build_trend_data(self, df: pd.DataFrame):
        """
        构建情绪时间趋势数据。

        默认使用当前 sentiment_comments.csv 中的全部评论。
        时间跨度小于等于 2 天时按小时统计；
        时间跨度超过 2 天时按日期统计。
        """
        if "comment_time" not in df.columns:
            print("缺少 comment_time 字段，无法绘制时间趋势图。")
            return pd.DataFrame(), ""

        trend_df = df.copy()
        trend_df["comment_time"] = pd.to_datetime(
            trend_df["comment_time"],
            errors="coerce"
        )

        trend_df = trend_df.dropna(subset=["comment_time"])

        if trend_df.empty:
            print("评论时间字段无法解析，无法绘制时间趋势图。")
            return pd.DataFrame(), ""

        min_time = trend_df["comment_time"].min()
        max_time = trend_df["comment_time"].max()
        time_span = max_time - min_time

        if time_span <= pd.Timedelta(days=2):
            trend_df["time_group"] = trend_df["comment_time"].dt.floor("h")
            group_type = "hour"
        else:
            trend_df["time_group"] = trend_df["comment_time"].dt.date
            group_type = "date"

        trend = (
            trend_df.groupby("time_group")
            .agg(
                avg_score=("sentiment_score", "mean"),
                comment_count=("sentiment_score", "count"),
                positive_count=("sentiment_type", lambda x: (x == "正向").sum()),
                neutral_count=("sentiment_type", lambda x: (x == "中立").sum()),
                negative_count=("sentiment_type", lambda x: (x == "负向").sum())
            )
            .reset_index()
        )

        trend["avg_score"] = trend["avg_score"].round(4)

        return trend, group_type

    def evaluate_trend(self, trend: pd.DataFrame):
        """
        自动评价情绪趋势。
        """
        if trend.empty or len(trend) <= 1:
            return (
                "时间分组数量不足，无法形成明显趋势。",
                "正负向变化不明显。"
            )

        first_score = trend["avg_score"].iloc[0]
        last_score = trend["avg_score"].iloc[-1]
        max_score = trend["avg_score"].max()
        min_score = trend["avg_score"].min()

        change = round(float(last_score - first_score), 4)
        fluctuation = round(float(max_score - min_score), 4)

        if change > 0.03:
            trend_comment = (
                f"整体情绪呈上升趋势，平均情感得分从 "
                f"{first_score:.4f} 上升到 {last_score:.4f}，"
                f"提升 {change:.4f}。"
            )
        elif change < -0.03:
            trend_comment = (
                f"整体情绪呈下降趋势，平均情感得分从 "
                f"{first_score:.4f} 下降到 {last_score:.4f}，"
                f"下降 {abs(change):.4f}。"
            )
        else:
            trend_comment = (
                f"整体情绪基本稳定，平均情感得分从 "
                f"{first_score:.4f} 变化到 {last_score:.4f}，"
                f"变化幅度较小。"
            )

        if fluctuation >= 0.35:
            fluctuation_comment = (
                f"不同时间段之间情感得分波动较明显，"
                f"最高与最低平均得分差值为 {fluctuation:.4f}。"
            )
        elif fluctuation >= 0.15:
            fluctuation_comment = (
                f"不同时间段之间存在一定情绪波动，"
                f"最高与最低平均得分差值为 {fluctuation:.4f}。"
            )
        else:
            fluctuation_comment = (
                f"不同时间段之间情绪波动较小，"
                f"最高与最低平均得分差值为 {fluctuation:.4f}。"
            )

        return trend_comment, fluctuation_comment

    def draw_sentiment_trend(self, df: pd.DataFrame, save_path: str):
        """
        绘制平均情感得分时间趋势图。
        """
        trend, group_type = self.build_trend_data(df)

        if trend.empty or len(trend) <= 1:
            print("时间分组数量不足，无法形成明显趋势图。")
            return pd.DataFrame(), "时间分组数量不足，无法形成明显趋势。", ""

        if group_type == "hour":
            x_label = "时间（按小时）"
            title = "图3  评论平均情感得分时间变化趋势"
            subtitle = "按小时统计平均情感得分"
        else:
            x_label = "日期"
            title = "图3  评论平均情感得分时间变化趋势"
            subtitle = "按日期统计平均情感得分"

        trend_comment, fluctuation_comment = self.evaluate_trend(trend)

        fig, ax = plt.subplots(figsize=(9.2, 5.6))

        ax.plot(
            trend["time_group"],
            trend["avg_score"],
            marker="o",
            markersize=5,
            linewidth=2.0,
            color=self.colors["main"],
            alpha=0.95
        )

        ax.fill_between(
            trend["time_group"],
            trend["avg_score"],
            [0] * len(trend),
            color=self.colors["main"],
            alpha=0.08
        )

        ax.axhline(
            0.6,
            color=self.colors["positive"],
            linestyle="--",
            linewidth=1.2,
            alpha=0.65
        )
        ax.axhline(
            0.4,
            color=self.colors["negative"],
            linestyle="--",
            linewidth=1.2,
            alpha=0.65
        )

        ax.text(
            trend["time_group"].iloc[-1],
            0.615,
            "正向阈值 0.6",
            fontsize=8.5,
            color=self.colors["positive"],
            ha="right"
        )
        ax.text(
            trend["time_group"].iloc[-1],
            0.415,
            "负向阈值 0.4",
            fontsize=8.5,
            color=self.colors["negative"],
            ha="right"
        )

        important_indices = {
            0,
            len(trend) - 1,
            int(trend["avg_score"].idxmax()),
            int(trend["avg_score"].idxmin())
        }

        for index in important_indices:
            row = trend.loc[index]
            ax.text(
                row["time_group"],
                row["avg_score"] + 0.035,
                f"{row['avg_score']:.2f}",
                ha="center",
                fontsize=9,
                color=self.colors["dark"]
            )

        ax.set_xlabel(x_label, fontsize=11)
        ax.set_ylabel("平均情感得分", fontsize=11)
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", rotation=35)

        self.style_axes(ax, grid_axis="y")
        self.add_title(ax, title, subtitle)

        self.save_figure(save_path)

        print("\n========== 情绪趋势评价 ==========")
        print(trend_comment)
        print(fluctuation_comment)

        return trend, trend_comment, fluctuation_comment

    def shorten_text(self, text: str, max_len: int = 24) -> str:
        """
        缩短评论文本，避免图表标签过长。
        """
        text = str(text).replace("\n", " ").strip()

        if len(text) <= max_len:
            return text

        return text[:max_len] + "..."

    def draw_influence_comments_bar(self, df: pd.DataFrame, save_path: str):
        """
        绘制高影响力评论排行图。

        影响力分数 = 点赞数 × 情绪极端程度。
        """
        if df.empty:
            print("没有评论数据，无法绘制高影响力评论排行图。")
            return pd.DataFrame()

        top_df = (
            df.sort_values(by="influence_score", ascending=False)
            .head(8)
            .copy()
        )

        if top_df.empty:
            print("没有足够数据生成高影响力评论排行图。")
            return pd.DataFrame()

        top_df["short_comment"] = top_df["clean_text"].apply(
            lambda x: self.shorten_text(x, max_len=24)
        )

        top_df = top_df.iloc[::-1]

        color_map = {
            "正向": self.colors["positive"],
            "中立": self.colors["neutral"],
            "负向": self.colors["negative"]
        }

        bar_colors = [
            color_map.get(sentiment_type, self.colors["gray"])
            for sentiment_type in top_df["sentiment_type"]
        ]

        fig, ax = plt.subplots(figsize=(9.6, 6.4))

        bars = ax.barh(
            top_df["short_comment"],
            top_df["influence_score"],
            color=bar_colors,
            alpha=0.9,
            height=0.66
        )

        max_score = top_df["influence_score"].max()
        ax.set_xlim(0, max_score * 1.25 if max_score > 0 else 1)

        for bar, (_, row) in zip(bars, top_df.iterrows()):
            width = bar.get_width()
            label = (
                f"{row['sentiment_type']} | "
                f"赞 {int(row['like_count'])} | "
                f"{width:.1f}"
            )

            ax.text(
                width + max_score * 0.025,
                bar.get_y() + bar.get_height() / 2,
                label,
                va="center",
                fontsize=9,
                color=self.colors["dark"]
            )

        ax.set_xlabel("影响力分数", fontsize=11)
        ax.set_ylabel("代表性评论", fontsize=11)

        self.style_axes(ax, grid_axis="x")
        self.add_title(
            ax,
            "图4  高影响力评论排行",
            "按 点赞数 × 情绪极端程度 排序，展示更可能影响舆论氛围的评论"
        )

        self.save_figure(save_path)

        return top_df.iloc[::-1]

    def get_wordcloud_stopwords(self):
        """
        词云专用停用词。
        """
        return {
            "的", "了", "是", "我", "你", "他", "她", "它",
            "我们", "你们", "他们", "这个", "那个", "就是",
            "什么", "不是", "还有", "然后", "因为", "所以",
            "但是", "如果", "还是", "没有", "现在", "可以",
            "可能", "应该", "怎么", "这样", "一样", "一些",
            "这种", "这些", "那些", "自己", "别人", "大家",
            "东西", "感觉", "真的", "哈哈", "哈哈哈", "doge",
            "笑哭", "视频", "评论", "内容", "时候", "知道",
            "觉得", "进行", "方面", "一下", "有点"
        }

    def generate_wordcloud(self, texts, save_path: str, title: str):
        """
        生成词云图，作为补充展示图。
        """
        stopwords = self.get_wordcloud_stopwords()
        words = []

        import jieba

        for text in texts:
            if pd.isna(text):
                continue

            seg_list = jieba.lcut(str(text))

            for word in seg_list:
                word = word.strip()

                if not word:
                    continue

                if len(word) <= 1:
                    continue

                if word.isdigit():
                    continue

                if word in stopwords:
                    continue

                words.append(word)

        if not words:
            print(f"{title} 没有足够词语生成词云。")
            return

        word_text = " ".join(words)

        wc = WordCloud(
            font_path=self.font_path,
            width=1200,
            height=780,
            background_color="white",
            max_words=160,
            collocations=False,
            prefer_horizontal=0.9
        ).generate(word_text)

        fig, ax = plt.subplots(figsize=(10, 7))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(
            title,
            fontsize=15,
            fontweight="bold",
            color=self.colors["dark"],
            pad=14
        )

        self.save_figure(save_path)

    def draw_wordclouds(self, df: pd.DataFrame, output_dir: str):
        """
        生成正向和负向词云，作为补充展示图。
        """
        positive_texts = df[df["sentiment_type"] == "正向"]["clean_text"]
        negative_texts = df[df["sentiment_type"] == "负向"]["clean_text"]

        self.generate_wordcloud(
            positive_texts,
            os.path.join(output_dir, "positive_wordcloud.png"),
            "正向评论词云"
        )

        self.generate_wordcloud(
            negative_texts,
            os.path.join(output_dir, "negative_wordcloud.png"),
            "负向评论词云"
        )

    def save_visualization_summary(
        self,
        df: pd.DataFrame,
        trend: pd.DataFrame,
        trend_comment: str,
        fluctuation_comment: str,
        influence_df: pd.DataFrame,
        save_path: str
    ):
        """
        保存可视化摘要。
        """
        lines = []
        lines.append("========== 可视化分析摘要 ==========\n\n")

        lines.append("一、基础信息\n")
        lines.append(f"参与可视化的评论数量：{len(df)} 条\n\n")

        lines.append("二、情感分布\n")
        distribution = (
            df["sentiment_type"]
            .value_counts()
            .reindex(["正向", "中立", "负向"], fill_value=0)
        )

        for sentiment_type, count in distribution.items():
            ratio = round(count / len(df) * 100, 2)
            lines.append(f"{sentiment_type}：{count} 条，占比 {ratio}%\n")

        lines.append("\n三、情绪时间趋势评价\n")
        lines.append(f"{trend_comment}\n")
        lines.append(f"{fluctuation_comment}\n")

        if trend is not None and not trend.empty:
            lines.append("\n趋势分组数据：\n")
            for _, row in trend.iterrows():
                lines.append(
                    f"{row['time_group']}："
                    f"平均得分 {row['avg_score']}，"
                    f"评论数 {int(row['comment_count'])}，"
                    f"正向 {int(row['positive_count'])}，"
                    f"中立 {int(row['neutral_count'])}，"
                    f"负向 {int(row['negative_count'])}\n"
                )

        lines.append("\n四、高影响力评论摘要\n")
        if influence_df is not None and not influence_df.empty:
            for _, row in influence_df.head(5).iterrows():
                lines.append(
                    f"- 类型：{row['sentiment_type']}；"
                    f"点赞数：{int(row['like_count'])}；"
                    f"影响力分数：{row['influence_score']}；"
                    f"评论：{row['clean_text']}\n"
                )
        else:
            lines.append("暂无高影响力评论结果。\n")

        lines.append("\n五、补充展示图说明\n")
        lines.append("sentiment_pie.png：情感分布环形图，用于直观展示正向、中立、负向比例。\n")
        lines.append("positive_wordcloud.png：正向评论词云，作为正向评论内容的补充展示。\n")
        lines.append("negative_wordcloud.png：负向评论词云，作为负向评论内容的补充展示。\n")
        lines.append("说明：词云仅作为展示补充，核心分析不依赖词云结果。\n")

        try:
            with open(save_path, "w", encoding="utf-8") as file:
                file.writelines(lines)

            print(f"可视化分析摘要已保存：{save_path}")
        except PermissionError:
            print(f"保存失败：{save_path} 可能正在被打开，请关闭后重试。")

    def run(self, sentiment_path: str, output_dir: str):
        """
        执行完整可视化流程。
        """
        self.ensure_output_dir(output_dir)
        df = self.load_sentiment_data(sentiment_path)

        print("\n正在生成稳定分析版可视化图表……")

        # 主分析图 1：情感类别柱状图
        self.draw_sentiment_type_bar(
            df,
            os.path.join(output_dir, "sentiment_type_bar.png")
        )

        # 补充展示图：情感分布环形图
        self.draw_sentiment_pie(
            df,
            os.path.join(output_dir, "sentiment_pie.png")
        )

        # 主分析图 2：情感得分分布
        self.draw_score_hist(
            df,
            os.path.join(output_dir, "sentiment_score_hist.png")
        )

        # 主分析图 3：情绪时间趋势
        trend, trend_comment, fluctuation_comment = (
            self.draw_sentiment_trend(
                df,
                os.path.join(output_dir, "sentiment_trend.png")
            )
        )

        # 主分析图 4：高影响力评论排行
        influence_df = self.draw_influence_comments_bar(
            df,
            os.path.join(output_dir, "influence_comments_bar.png")
        )

        # 补充展示图：正向/负向词云
        self.draw_wordclouds(df, output_dir)

        self.save_visualization_summary(
            df=df,
            trend=trend,
            trend_comment=trend_comment,
            fluctuation_comment=fluctuation_comment,
            influence_df=influence_df,
            save_path=os.path.join(output_dir, "visualization_summary.txt")
        )

        print("\n稳定分析版可视化图表生成完成。")


if __name__ == "__main__":
    sentiment_file = os.path.join("data", "sentiment_comments.csv")
    output_folder = "output"

    visualizer = CommentVisualizer()
    visualizer.run(sentiment_file, output_folder)