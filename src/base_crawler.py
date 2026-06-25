from abc import ABC, abstractmethod


class BaseCommentCrawler(ABC):
    """
    评论爬虫基类。

    不同平台的评论接口、字段名称、分页方式可能不同，
    但进入分析模块之前，都需要转换成统一的评论数据格式。
    """

    @abstractmethod
    def fetch_comments(self, content_id: str, max_pages: int):
        """
        获取评论数据。

        :param content_id: 视频、笔记或内容 ID
        :param max_pages: 最大爬取页数
        :return: 标准化后的评论列表
        """
        pass

    @abstractmethod
    def normalize_comment(self, raw_comment: dict):
        """
        将平台原始评论字段转换成统一格式。

        统一字段包括：
        platform, content_id, user_name, comment_text,
        like_count, comment_time, reply_count
        """
        pass