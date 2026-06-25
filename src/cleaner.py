import os
import re
from datetime import datetime

import pandas as pd


class CommentCleaner:
    """
    评论数据清洗类。

    功能：
    1. 读取 raw_comments.csv 原始评论
    2. 删除空评论
    3. 删除重复评论
    4. 去除 @用户、表情符号、链接、多余空格
    5. 过滤低信息评论
    6. 保存 clean_comments.csv
    """

    def __init__(self):
        self.low_value_words = {
            "哈哈", "哈哈哈", "哈哈哈哈", "哈哈哈哈哈",
            "666", "6", "啊", "哦", "嗯", "额",
            "来了", "来啦", "第一", "前排", "打卡",
            "沙发", "支持", "路过", "看看", "围观",
            "牛", "牛逼", "nb", "NB", "好", "顶",
            "笑死", "笑死我了"
        }

        self.emotion_words = {
            "喜欢", "讨厌", "感动", "震撼", "破防", "真实",
            "离谱", "无语", "失望", "支持", "反对", "认可",
            "心疼", "难受", "开心", "愤怒", "生气", "尴尬",
            "好看", "难看", "精彩", "厉害", "佩服", "赞同",
            "不赞同", "有道理", "没道理", "恶心", "舒服",
            "治愈", "焦虑", "压力", "热血", "燃", "哭了"
        }

    def load_data(self, raw_path: str) -> pd.DataFrame:
        """
        读取原始评论数据。
        """
        if not os.path.exists(raw_path):
            raise FileNotFoundError(f"未找到原始评论文件：{raw_path}")

        df = pd.read_csv(raw_path, encoding="utf-8-sig")
        print(f"成功读取原始评论数据：{raw_path}")
        print(f"原始评论数量：{len(df)} 条")

        return df

    def remove_mentions(self, text: str) -> str:
        """
        去除评论中的 @用户。

        例如：
        @张三 这个观点很真实  ->  这个观点很真实
        @李四 快来看          ->  快来看
        """
        text = re.sub(r"@\S+", "", text)
        return text.strip()

    def remove_url(self, text: str) -> str:
        """
        去除评论中的链接。
        """
        text = re.sub(r"http[s]?://\S+", "", text)
        text = re.sub(r"www\.\S+", "", text)
        return text.strip()

    def remove_emoji_and_symbols(self, text: str) -> str:
        """
        去除大部分表情符号和特殊符号，保留中文、英文、数字和常见标点。
        """
        text = re.sub(
            r"[^\u4e00-\u9fa5a-zA-Z0-9，。！？、,.!?：:；;（）()《》“”\"' \-]",
            "",
            text
        )
        return text.strip()

    def normalize_space(self, text: str) -> str:
        """
        合并多余空格。
        """
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def clean_text(self, text: str) -> str:
        """
        对单条评论文本进行基础清洗。
        """
        if pd.isna(text):
            return ""

        text = str(text).strip()
        text = self.remove_url(text)
        text = self.remove_mentions(text)
        text = self.remove_emoji_and_symbols(text)
        text = self.normalize_space(text)

        return text

    def has_emotion_word(self, text: str) -> bool:
        """
        判断短评论中是否包含情绪关键词。

        例如：
        “太震撼了”虽然短，但应该保留。
        “666”虽然短，但信息价值低，可以过滤。
        """
        for word in self.emotion_words:
            if word in text:
                return True
        return False

    def is_low_value_comment(self, text: str) -> bool:
        """
        判断是否为低信息评论。

        过滤：
        1. 空评论
        2. 纯数字
        3. 纯低价值词
        4. 过短且没有情绪词的评论
        """
        if not text:
            return True

        text_no_space = text.replace(" ", "")

        if not text_no_space:
            return True

        if text_no_space in self.low_value_words:
            return True

        if text_no_space.isdigit():
            return True

        # 过短但有情绪词的评论保留，例如“太震撼了”“好感动”
        if len(text_no_space) <= 4 and not self.has_emotion_word(text_no_space):
            return True

        return False

    def clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        清洗整个 DataFrame。
        """
        if "comment_text" not in df.columns:
            raise ValueError("数据中缺少 comment_text 字段，无法清洗。")

        before_count = len(df)

        # 删除完全空的评论
        df = df.dropna(subset=["comment_text"]).copy()

        # 优先按 rpid 去重，如果没有 rpid，则按评论内容去重
        if "rpid" in df.columns:
            df = df.drop_duplicates(subset=["rpid"])
        else:
            df = df.drop_duplicates(subset=["comment_text"])

        after_duplicate_count = len(df)

        # 保留原始评论内容
        df["original_comment_text"] = df["comment_text"]

        # 清洗文本
        df["clean_text"] = df["comment_text"].apply(self.clean_text)

        # 过滤低信息评论
        df["is_low_value"] = df["clean_text"].apply(self.is_low_value_comment)
        df_clean = df[df["is_low_value"] == False].copy()

        # 删除辅助列
        df_clean = df_clean.drop(columns=["is_low_value"])

        after_clean_count = len(df_clean)

        print("\n========== 数据清洗统计 ==========")
        print(f"原始评论数量：{before_count} 条")
        print(f"去重后评论数量：{after_duplicate_count} 条")
        print(f"清洗后有效评论数量：{after_clean_count} 条")
        print(f"过滤低信息评论数量：{after_duplicate_count - after_clean_count} 条")

        return df_clean

    def save_clean_data(self, df: pd.DataFrame, save_path: str) -> str:
        """
        保存清洗后的评论数据。
        """
        folder = os.path.dirname(save_path)

        if folder and not os.path.exists(folder):
            os.makedirs(folder)

        try:
            df.to_csv(save_path, index=False, encoding="utf-8-sig")
            final_path = save_path
        except PermissionError:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            name, ext = os.path.splitext(save_path)
            final_path = f"{name}_{timestamp}{ext}"
            df.to_csv(final_path, index=False, encoding="utf-8-sig")
            print("clean_comments.csv 可能正在被 Excel/WPS 占用。")
            print(f"已自动另存为：{final_path}")

        print(f"清洗后评论数据已保存到：{final_path}")
        return final_path

    def run(self, raw_path: str, clean_path: str) -> str:
        """
        执行完整清洗流程。
        """
        df_raw = self.load_data(raw_path)
        df_clean = self.clean_dataframe(df_raw)
        final_path = self.save_clean_data(df_clean, clean_path)

        print("\n前 5 条清洗结果预览：")
        preview_columns = ["original_comment_text", "clean_text"]

        for index, row in df_clean[preview_columns].head(5).iterrows():
            print(f"\n原评论：{row['original_comment_text']}")
            print(f"清洗后：{row['clean_text']}")

        return final_path


if __name__ == "__main__":
    raw_file = os.path.join("data", "raw_comments.csv")
    clean_file = os.path.join("data", "clean_comments.csv")

    cleaner = CommentCleaner()
    cleaner.run(raw_file, clean_file)