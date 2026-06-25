from base_crawler import BaseCommentCrawler


class XiaohongshuCommentCrawler(BaseCommentCrawler):
    """
    小红书评论爬虫预留类。

    当前期末项目以 B 站作为实验平台，
    该类用于体现后续多平台扩展能力。
    """

    def fetch_comments(self, content_id: str, max_pages: int):
        raise NotImplementedError("后续可接入小红书评论接口")

    def normalize_comment(self, raw_comment: dict):
        raise NotImplementedError("后续可实现小红书字段标准化")


class DouyinCommentCrawler(BaseCommentCrawler):
    """
    抖音评论爬虫预留类。

    后续只需要实现该类，就可以复用已有的数据清洗、
    情感分析、争议度计算和可视化模块。
    """

    def fetch_comments(self, content_id: str, max_pages: int):
        raise NotImplementedError("后续可接入抖音评论接口")

    def normalize_comment(self, raw_comment: dict):
        raise NotImplementedError("后续可实现抖音字段标准化")