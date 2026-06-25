import json
import os
from typing import Dict, Any

from dotenv import load_dotenv
from google import genai


class AISentimentReviewer:
    """
    Gemini AI 情感复核模块。

    作用：
    1. 不替代 SnowNLP 的全部分析
    2. 只复核高影响力评论、转折句、反讽句、质疑句等关键评论
    3. 返回固定 JSON 字段，方便程序稳定读取
    """

    def __init__(self):
        load_dotenv()

        self.enabled = os.getenv("AI_REVIEW_ENABLED", "false").lower() == "true"
        self.provider = os.getenv("AI_PROVIDER", "gemini").lower()
        self.api_key = os.getenv("GEMINI_API_KEY", "").strip()
        self.model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()
        self.max_review_count = int(os.getenv("AI_REVIEW_MAX_COUNT", "40"))

        if self.enabled and self.provider != "gemini":
            print("当前 ai_sentiment_reviewer.py 使用 Gemini，请将 AI_PROVIDER 设置为 gemini。")
            self.enabled = False

        if self.enabled and not self.api_key:
            print("警告：AI_REVIEW_ENABLED=true，但未检测到 GEMINI_API_KEY，AI复核将自动关闭。")
            self.enabled = False

        self.client = genai.Client(api_key=self.api_key) if self.enabled else None

    def is_available(self) -> bool:
        """
        判断 AI 复核是否可用。
        """
        return self.enabled and self.client is not None

    def review_comment(
        self,
        comment_text: str,
        snow_score: float,
        rule_score: float,
        rule_type: str,
        like_count: int = 0
    ) -> Dict[str, Any]:
        """
        使用 Gemini API 复核单条评论情感。

        返回字段：
        sentiment_type: 正向 / 中立 / 负向
        sentiment_score: 0-1
        reason: 判断理由
        confidence: 0-1
        """
        if not self.is_available():
            return self._fallback_result(
                rule_type,
                rule_score,
                "AI复核未启用，保留规则判断结果"
            )

        prompt = f"""
你是一个中文网络评论情感分析助手。
请判断下面这条中文网络评论的真实情感倾向。

判断要求：
1. 不要只看单个正向词或负向词，要结合完整语境。
2. 对“但是、不过、可是、然而、却、只是”等转折句，要重点看转折后的真实态度。
3. 对反讽、调侃、阴阳怪气、营销质疑、反问句要特别谨慎。
4. “不否认是好剧，但你说没营销就笑了”这类表达应理解为质疑/负向，而不是正向。
5. 如果评论只是陈述事实且态度不明显，判为中立。
6. sentiment_score 范围是 0 到 1，越接近 1 越正向，越接近 0 越负向。

评论内容：
{comment_text}

SnowNLP原始得分：{snow_score}
规则修正后得分：{rule_score}
规则修正后类型：{rule_type}
点赞数：{like_count}

请只输出 JSON，不要输出其他解释文字。
"""

        response_schema = {
            "type": "object",
            "properties": {
                "sentiment_type": {
                    "type": "string",
                    "enum": ["正向", "中立", "负向"]
                },
                "sentiment_score": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1
                },
                "reason": {
                    "type": "string"
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1
                }
            },
            "required": [
                "sentiment_type",
                "sentiment_score",
                "reason",
                "confidence"
            ]
        }

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_json_schema": response_schema,
                    "temperature": 0
                }
            )

            content = response.text
            result = json.loads(content)

            return {
                "sentiment_type": result["sentiment_type"],
                "sentiment_score": round(float(result["sentiment_score"]), 4),
                "reason": result["reason"],
                "confidence": round(float(result["confidence"]), 4),
                "success": True
            }

        except Exception as error:
            return self._fallback_result(
                rule_type,
                rule_score,
                f"AI复核失败，保留规则判断结果。错误信息：{error}"
            )

    def _fallback_result(
        self,
        rule_type: str,
        rule_score: float,
        reason: str
    ) -> Dict[str, Any]:
        """
        AI 不可用或失败时，保留规则判断结果。
        """
        return {
            "sentiment_type": rule_type,
            "sentiment_score": round(float(rule_score), 4),
            "reason": reason,
            "confidence": 0,
            "success": False
        }