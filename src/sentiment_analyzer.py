import os
import time

import numpy as np
import pandas as pd
from snownlp import SnowNLP

from ai_sentiment_reviewer import AISentimentReviewer


class SentimentAnalyzer:
    """
    评论情感分析类：SnowNLP + 规则修正 + AI关键评论复核。

    优化点：
    1. 不再显示“AI复核：False”这种容易误解的表达
    2. 增加“判断来源”和“AI复核状态”
    3. 强制复核高影响力 Top8 和点赞 Top8 评论
    4. 增加社会焦虑、黑色幽默、就业压力类规则
    """

    def __init__(self):
        self.positive_threshold = 0.6
        self.negative_threshold = 0.4

        self.positive_words = {
            "喜欢", "支持", "赞同", "有道理", "真实", "感动", "震撼",
            "好看", "精彩", "厉害", "佩服", "治愈", "温暖", "不错",
            "舒服", "热血", "燃", "清楚", "明白", "优秀",
            "好棒", "太棒", "赞", "值得", "希望", "加油", "上大分",
            "太美了", "看不够", "推荐"
        }

        self.negative_words = {
            "为什么非得", "非得", "凭什么", "没必要", "不理解",
            "离谱", "无语", "失望", "讨厌", "恶心", "尴尬",
            "反对", "不赞同", "没道理", "质疑", "讽刺",
            "不公平", "荒谬", "可笑", "麻木", "难绷"
        }

        self.anxiety_words = {
            "焦虑", "压力", "难受", "痛苦", "绝望", "崩溃",
            "太累", "累", "卷", "压抑", "害怕", "害怕死了",
            "恐惧", "担心", "完蛋", "大哭", "哭死", "吓人",
            "吓死", "可怕", "慌", "恐慌", "绷不住"
        }

        self.comfort_words = {
            "别想", "别怕", "别难过", "没事", "加油", "吃碗",
            "休息一下", "放松", "慢慢来", "会好的", "抱抱",
            "别焦虑", "别急", "不要怕", "都会过去"
        }

        self.question_negative_patterns = {
            "为什么", "凭什么", "非得", "一定要", "有必要吗",
            "真的需要吗", "为什么要", "谁规定", "难道"
        }

        self.turning_words = {
            "但是", "但", "不过", "可是", "然而", "却", "只是"
        }

        self.sarcasm_words = {
            "就笑了", "笑了", "笑死", "你说呢", "真的假的",
            "不会真有人", "别洗", "洗白", "营销", "硬吹",
            "尬吹", "吹过了", "投了不少", "买热搜", "水军",
            "尴尬", "离谱", "无语", "黑色幽默", "讽刺",
            "荒诞", "魔幻", "现实太魔幻"
        }

        self.doubt_words = {
            "不否认", "不是说", "不代表", "说没", "没营销",
            "不觉得", "不理解", "不认同", "不认可",
            "有问题", "有点问题"
        }

        # 社会焦虑 / 就业压力 / 黑色幽默类表达
        self.social_anxiety_words = {
            "黑色幽默", "惨淡", "就业", "就业情况", "就业压力",
            "大学毕业", "毕业后", "毕业即失业", "全社会",
            "高考气氛", "烘托", "重视", "对比来看",
            "现实", "社会", "荒诞", "魔幻"
        }

        self.ai_reviewer = AISentimentReviewer()

    def load_clean_data(self, clean_path: str) -> pd.DataFrame:
        if not os.path.exists(clean_path):
            raise FileNotFoundError(f"未找到清洗后评论文件：{clean_path}")

        df = pd.read_csv(clean_path, encoding="utf-8-sig")

        if "clean_text" not in df.columns:
            raise ValueError("数据中缺少 clean_text 字段，无法进行情感分析。")

        print(f"成功读取清洗后评论数据：{clean_path}")
        print(f"待分析有效评论数量：{len(df)} 条")

        return df

    def get_base_score(self, text: str) -> float:
        if pd.isna(text) or str(text).strip() == "":
            return 0.5

        try:
            return float(SnowNLP(str(text)).sentiments)
        except Exception:
            return 0.5

    def correct_score_by_rules(self, text: str, base_score: float):
        """
        规则修正：
        1. 转折句后半句优先
        2. 讽刺、质疑、焦虑类表达修正
        3. 社会焦虑 / 黑色幽默 / 就业压力类表达修正
        """
        text = str(text).strip()
        score = base_score
        reasons = []

        turning_part = ""

        for word in self.turning_words:
            if word in text:
                parts = text.split(word, 1)
                if len(parts) == 2 and parts[1].strip():
                    turning_part = parts[1].strip()
                    reasons.append("检测到转折句，优先分析转折后内容")
                    break

        focus_text = turning_part if turning_part else text

        if any(word in focus_text for word in self.comfort_words):
            if score < 0.65:
                score = 0.65
                reasons.append("安慰/鼓励类表达修正为偏正向")

        if any(word in focus_text for word in self.positive_words):
            if score < 0.7:
                score = 0.7
                reasons.append("正向关键词修正")

        if any(word in focus_text for word in self.negative_words):
            if score > 0.3:
                score = 0.3
                reasons.append("负向关键词修正")

        if any(word in focus_text for word in self.anxiety_words):
            if score > 0.2:
                score = 0.2
                reasons.append("焦虑/恐惧类表达修正")

        if "？" in focus_text or "?" in focus_text:
            if any(pattern in focus_text for pattern in self.question_negative_patterns):
                if score > 0.35:
                    score = 0.35
                    reasons.append("反问/质疑句式修正")

        if any(word in focus_text for word in self.sarcasm_words):
            if score > 0.25:
                score = 0.25
                reasons.append("讽刺/质疑类表达修正")

        if any(word in focus_text for word in self.doubt_words):
            if score > 0.35:
                score = 0.35
                reasons.append("否定/质疑类表达修正")

        # 社会焦虑 / 黑色幽默 / 就业压力类规则
        has_social_anxiety = any(word in text for word in self.social_anxiety_words)

        high_exam_and_job = ("高考" in text and "就业" in text)
        graduate_and_bad_job = (
            ("大学毕业" in text or "毕业后" in text)
            and ("惨淡" in text or "就业" in text or "失业" in text)
        )

        black_humor_signal = "黑色幽默" in text

        if has_social_anxiety or high_exam_and_job or graduate_and_bad_job or black_humor_signal:
            if score > 0.3:
                score = 0.3
            reasons.append("社会焦虑/黑色幽默/就业压力表达修正为偏负向")

        if turning_part:
            turning_negative_signals = (
                self.negative_words
                | self.anxiety_words
                | self.sarcasm_words
                | self.doubt_words
                | self.social_anxiety_words
            )

            if any(word in turning_part for word in turning_negative_signals):
                if score > 0.25:
                    score = 0.25
                reasons.append("转折后内容带有负向/质疑信号，修正为负向")

        if not reasons:
            reasons.append("SnowNLP原始判断")

        return round(float(score), 4), "；".join(reasons)

    def classify_sentiment(self, score: float) -> str:
        if score > self.positive_threshold:
            return "正向"
        if score < self.negative_threshold:
            return "负向"
        return "中立"

    def analyze_single_text(self, text: str):
        snow_score = self.get_base_score(text)
        rule_score, rule_reason = self.correct_score_by_rules(text, snow_score)
        rule_type = self.classify_sentiment(rule_score)

        return (
            round(float(snow_score), 4),
            rule_score,
            rule_type,
            rule_reason
        )

    def analyze_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        print("\n正在进行 SnowNLP 与规则情感分析，请稍等……")

        df = df.copy()

        results = df["clean_text"].apply(self.analyze_single_text)

        df["snow_score"] = results.apply(lambda item: item[0])
        df["rule_score"] = results.apply(lambda item: item[1])
        df["rule_sentiment_type"] = results.apply(lambda item: item[2])
        df["rule_reason"] = results.apply(lambda item: item[3])

        df["sentiment_score"] = df["rule_score"]
        df["sentiment_type"] = df["rule_sentiment_type"]
        df["sentiment_reason"] = df["rule_reason"]

        if "like_count" in df.columns:
            df["like_count"] = pd.to_numeric(
                df["like_count"],
                errors="coerce"
            ).fillna(0)
        else:
            df["like_count"] = 0

        df["emotion_extreme_degree"] = abs(df["sentiment_score"] - 0.5)
        df["influence_score"] = (
            df["like_count"] * df["emotion_extreme_degree"]
        ).round(4)

        df["ai_reviewed"] = False
        df["ai_sentiment_type"] = ""
        df["ai_sentiment_score"] = np.nan
        df["ai_reason"] = ""
        df["ai_confidence"] = np.nan

        # 新增更清楚的显示字段
        df["judgment_source"] = "SnowNLP + 规则"
        df["ai_review_status"] = "未进入AI复核队列"

        print("SnowNLP 与规则分析完成。")

        return df

    def need_ai_review(self, row: pd.Series) -> bool:
        text = str(row.get("clean_text", ""))

        like_count = float(row.get("like_count", 0))
        influence_score = float(row.get("influence_score", 0))
        snow_score = float(row.get("snow_score", 0.5))
        rule_score = float(row.get("rule_score", 0.5))
        rule_reason = str(row.get("rule_reason", ""))

        has_turning = any(word in text for word in self.turning_words)
        has_question = ("？" in text) or ("?" in text)
        has_sarcasm = any(word in text for word in self.sarcasm_words)
        has_doubt = any(word in text for word in self.doubt_words)
        has_social_anxiety = any(word in text for word in self.social_anxiety_words)

        high_exam_and_job = ("高考" in text and "就业" in text)
        graduate_and_bad_job = (
            ("大学毕业" in text or "毕业后" in text)
            and ("惨淡" in text or "就业" in text or "失业" in text)
        )

        snow_rule_gap = abs(snow_score - rule_score)

        if like_count >= 50:
            return True

        if influence_score >= 15:
            return True

        if has_turning or has_question or has_sarcasm or has_doubt:
            return True

        if has_social_anxiety or high_exam_and_job or graduate_and_bad_job:
            return True

        if snow_rule_gap >= 0.4:
            return True

        if "SnowNLP原始判断" not in rule_reason and like_count >= 10:
            return True

        return False

    def get_ai_candidate_indices(self, df: pd.DataFrame):
        """
        构建 AI 复核队列。

        规则：
        1. 强制加入高影响力 Top8
        2. 强制加入点赞 Top8
        3. 加入复杂表达候选评论
        4. 去重
        5. 按影响力排序，限制总数量
        """
        max_count = self.ai_reviewer.max_review_count

        force_influence_indices = (
            df.sort_values(by="influence_score", ascending=False)
            .head(8)
            .index
            .tolist()
        )

        force_like_indices = (
            df.sort_values(by="like_count", ascending=False)
            .head(8)
            .index
            .tolist()
        )

        complex_indices = df[df.apply(self.need_ai_review, axis=1)].index.tolist()

        candidate_indices = []

        for index in force_influence_indices + force_like_indices + complex_indices:
            if index not in candidate_indices:
                candidate_indices.append(index)

        candidate_df = df.loc[candidate_indices].copy()

        candidate_df = candidate_df.sort_values(
            by=["influence_score", "like_count"],
            ascending=False
        )

        final_indices = candidate_df.head(max_count).index.tolist()

        return final_indices

    def apply_ai_review(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.ai_reviewer.is_available():
            print("AI复核未启用或不可用，跳过 AI 复核步骤。")
            return df

        print("\n正在筛选需要 AI 复核的关键评论……")

        df = df.copy()
        candidate_indices = self.get_ai_candidate_indices(df)

        if not candidate_indices:
            print("没有需要 AI 复核的评论。")
            return df

        print(f"AI复核候选评论数量：{len(candidate_indices)} 条")

        for count, index in enumerate(candidate_indices, start=1):
            row = df.loc[index]

            print(f"正在 AI 复核第 {count}/{len(candidate_indices)} 条评论……")

            df.loc[index, "ai_review_status"] = "已进入AI复核队列"

            review_result = self.ai_reviewer.review_comment(
                comment_text=row["clean_text"],
                snow_score=row["snow_score"],
                rule_score=row["rule_score"],
                rule_type=row["rule_sentiment_type"],
                like_count=int(row["like_count"])
            )

            df.loc[index, "ai_sentiment_type"] = review_result["sentiment_type"]
            df.loc[index, "ai_sentiment_score"] = review_result["sentiment_score"]
            df.loc[index, "ai_reason"] = review_result["reason"]
            df.loc[index, "ai_confidence"] = review_result["confidence"]

            if review_result["success"]:
                df.loc[index, "ai_reviewed"] = True
                df.loc[index, "judgment_source"] = "Gemini AI复核"
                df.loc[index, "ai_review_status"] = "已完成AI复核"

                df.loc[index, "sentiment_type"] = review_result["sentiment_type"]
                df.loc[index, "sentiment_score"] = review_result["sentiment_score"]
                df.loc[index, "sentiment_reason"] = (
                    "AI复核："
                    + review_result["reason"]
                )
            else:
                df.loc[index, "ai_reviewed"] = False
                df.loc[index, "judgment_source"] = "SnowNLP + 规则"
                df.loc[index, "ai_review_status"] = "AI复核失败，保留规则判断"

            time.sleep(0.2)

        df["emotion_extreme_degree"] = abs(df["sentiment_score"] - 0.5)
        df["influence_score"] = (
            df["like_count"] * df["emotion_extreme_degree"]
        ).round(4)

        print("AI 关键评论复核完成。")

        return df

    def get_sentiment_distribution(self, df: pd.DataFrame) -> pd.DataFrame:
        total = len(df)

        distribution = (
            df["sentiment_type"]
            .value_counts()
            .reindex(["正向", "中立", "负向"], fill_value=0)
            .reset_index()
        )

        distribution.columns = ["sentiment_type", "count"]
        distribution["ratio"] = (
            distribution["count"] / total * 100
        ).round(2)

        return distribution

    def calculate_dispute_index(self, distribution: pd.DataFrame) -> float:
        positive_ratio = 0
        negative_ratio = 0

        for _, row in distribution.iterrows():
            if row["sentiment_type"] == "正向":
                positive_ratio = row["ratio"]
            elif row["sentiment_type"] == "负向":
                negative_ratio = row["ratio"]

        dispute_index = 1 - abs(positive_ratio - negative_ratio) / 100
        return round(dispute_index, 4)

    def get_dispute_level(self, dispute_index: float) -> str:
        if dispute_index >= 0.85:
            return "争议较高，评论区正负观点接近，存在明显分歧"
        if dispute_index >= 0.65:
            return "存在一定争议，评论区并非完全一边倒"
        return "争议较低，评论区观点相对集中"

    def get_representative_comments(self, df: pd.DataFrame) -> dict:
        representatives = {}

        if df.empty:
            return representatives

        representatives["most_positive"] = df.sort_values(
            by="sentiment_score",
            ascending=False
        ).head(3)

        representatives["most_negative"] = df.sort_values(
            by="sentiment_score",
            ascending=True
        ).head(3)

        representatives["high_like_positive"] = (
            df[df["sentiment_type"] == "正向"]
            .sort_values(by="like_count", ascending=False)
            .head(3)
        )

        representatives["high_like_negative"] = (
            df[df["sentiment_type"] == "负向"]
            .sort_values(by="like_count", ascending=False)
            .head(3)
        )

        representatives["high_influence"] = df.sort_values(
            by="influence_score",
            ascending=False
        ).head(5)

        return representatives

    def save_analyzed_data(self, df: pd.DataFrame, save_path: str) -> str:
        folder = os.path.dirname(save_path)

        if folder and not os.path.exists(folder):
            os.makedirs(folder)

        try:
            df.to_csv(save_path, index=False, encoding="utf-8-sig")
        except PermissionError:
            print(f"保存失败：{save_path} 可能正在被 Excel/WPS 打开。")
            print("请关闭该 CSV 文件后重新运行程序。")
            return ""

        print(f"情感分析结果已保存到：{save_path}")
        return save_path

    def save_summary_report(
        self,
        df: pd.DataFrame,
        distribution: pd.DataFrame,
        dispute_index: float,
        representatives: dict,
        save_path: str
    ) -> str:
        folder = os.path.dirname(save_path)

        if folder and not os.path.exists(folder):
            os.makedirs(folder)

        dispute_level = self.get_dispute_level(dispute_index)
        total = len(df)
        avg_score = round(float(np.mean(df["sentiment_score"])), 4)
        ai_review_count = int(df["ai_reviewed"].sum()) if "ai_reviewed" in df.columns else 0

        lines = []
        lines.append("========== B站评论情感分析摘要 ==========\n")
        lines.append(f"有效评论数量：{total} 条\n")
        lines.append(f"平均情感得分：{avg_score}\n")
        lines.append(f"争议度指数：{dispute_index}\n")
        lines.append(f"争议度解释：{dispute_level}\n")
        lines.append(f"AI复核完成数量：{ai_review_count} 条\n\n")

        lines.append("一、情感分布统计\n")
        for _, row in distribution.iterrows():
            lines.append(
                f"{row['sentiment_type']}："
                f"{int(row['count'])} 条，"
                f"占比 {row['ratio']}%\n"
            )

        lines.append("\n二、代表性评论\n")

        def append_comment_group(title, group_df):
            lines.append(f"\n【{title}】\n")

            if group_df.empty:
                lines.append("暂无对应评论。\n")
                return

            for _, row in group_df.iterrows():
                lines.append(
                    f"- SnowNLP原始得分：{row['snow_score']}，"
                    f"规则得分：{row['rule_score']}，"
                    f"最终得分：{row['sentiment_score']}，"
                    f"类型：{row['sentiment_type']}，"
                    f"判断来源：{row.get('judgment_source', 'SnowNLP + 规则')}，"
                    f"AI复核状态：{row.get('ai_review_status', '未进入AI复核队列')}，"
                    f"原因：{row['sentiment_reason']}，"
                    f"点赞：{int(row['like_count'])}，"
                    f"评论：{row['clean_text']}\n"
                )

        append_comment_group(
            "最正向评论 Top3",
            representatives.get("most_positive", pd.DataFrame())
        )

        append_comment_group(
            "最负向评论 Top3",
            representatives.get("most_negative", pd.DataFrame())
        )

        append_comment_group(
            "高赞正向评论 Top3",
            representatives.get("high_like_positive", pd.DataFrame())
        )

        append_comment_group(
            "高赞负向评论 Top3",
            representatives.get("high_like_negative", pd.DataFrame())
        )

        append_comment_group(
            "高影响力评论 Top5",
            representatives.get("high_influence", pd.DataFrame())
        )

        try:
            with open(save_path, "w", encoding="utf-8") as file:
                file.writelines(lines)
        except PermissionError:
            print(f"保存失败：{save_path} 可能正在被打开。")
            print("请关闭该文件后重新运行程序。")
            return ""

        print(f"情感分析摘要已保存到：{save_path}")
        return save_path

    def print_summary(
        self,
        distribution: pd.DataFrame,
        dispute_index: float,
        representatives: dict,
        df: pd.DataFrame
    ):
        print("\n========== 情感分析统计 ==========")

        for _, row in distribution.iterrows():
            print(
                f"{row['sentiment_type']}："
                f"{int(row['count'])} 条，"
                f"占比 {row['ratio']}%"
            )

        dispute_level = self.get_dispute_level(dispute_index)

        print(f"\n争议度指数：{dispute_index}")
        print(f"争议度解释：{dispute_level}")

        if "ai_reviewed" in df.columns:
            print(f"AI复核完成数量：{int(df['ai_reviewed'].sum())} 条")

        print("\n========== 代表性评论预览 ==========")

        high_influence = representatives.get("high_influence", pd.DataFrame())

        if high_influence.empty:
            print("暂无高影响力评论。")
            return

        for _, row in high_influence.head(5).iterrows():
            print("\n高影响力评论：")
            print(f"评论：{row['clean_text']}")
            print(f"SnowNLP原始得分：{row['snow_score']}")
            print(f"规则得分：{row['rule_score']}")
            print(f"最终情感得分：{row['sentiment_score']}")
            print(f"情感类型：{row['sentiment_type']}")
            print(f"修正原因：{row['sentiment_reason']}")
            print(f"判断来源：{row.get('judgment_source', 'SnowNLP + 规则')}")
            print(f"AI复核状态：{row.get('ai_review_status', '未进入AI复核队列')}")
            print(f"点赞数：{int(row['like_count'])}")
            print(f"影响力分数：{row['influence_score']}")

    def run(
        self,
        clean_path: str,
        sentiment_save_path: str,
        summary_save_path: str
    ):
        df_clean = self.load_clean_data(clean_path)

        df_analyzed = self.analyze_dataframe(df_clean)

        df_analyzed = self.apply_ai_review(df_analyzed)

        distribution = self.get_sentiment_distribution(df_analyzed)
        dispute_index = self.calculate_dispute_index(distribution)
        representatives = self.get_representative_comments(df_analyzed)

        self.save_analyzed_data(df_analyzed, sentiment_save_path)

        self.save_summary_report(
            df=df_analyzed,
            distribution=distribution,
            dispute_index=dispute_index,
            representatives=representatives,
            save_path=summary_save_path
        )

        self.print_summary(
            distribution=distribution,
            dispute_index=dispute_index,
            representatives=representatives,
            df=df_analyzed
        )


if __name__ == "__main__":
    clean_file = os.path.join("data", "clean_comments.csv")
    sentiment_file = os.path.join("data", "sentiment_comments.csv")
    summary_file = os.path.join("output", "sentiment_summary.txt")

    analyzer = SentimentAnalyzer()
    analyzer.run(clean_file, sentiment_file, summary_file)