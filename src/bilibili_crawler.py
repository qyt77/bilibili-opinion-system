import os
import re
from datetime import datetime

import pandas as pd
import requests

from base_crawler import BaseCommentCrawler


class BilibiliCommentCrawler(BaseCommentCrawler):
    """
    B站评论爬虫类。

    当前项目真实实现 B站评论采集。
    该类负责：
    1. 提取 BV 号
    2. 获取视频 aid
    3. 按页爬取评论
    4. 统一评论字段
    5. 保存原始评论数据

    注意：
    main.py 才是主程序入口。
    bilibili_crawler.py 主要作为爬虫模块被 main.py 调用。
    """

    def __init__(self):
        self.platform = "bilibili"
        self.video_info_url = "https://api.bilibili.com/x/web-interface/view"
        self.comment_main_url = "https://api.bilibili.com/x/v2/reply/main"
        self.comment_old_url = "https://api.bilibili.com/x/v2/reply"

        # B站评论接口每页通常取 20 条
        self.page_size = 20

        self.headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Referer": "https://www.bilibili.com/"
        }

    def extract_bvid(self, user_input: str) -> str:
        """
        从用户输入中提取 BV 号。

        支持：
        1. 直接输入 BV 号
        2. 输入完整链接
        3. 输入“标题 + 链接”的混合文本

        例如：
        BV1xxxxxxx
        https://www.bilibili.com/video/BV1xxxxxxx/
        【标题】https://www.bilibili.com/video/BV1xxxxxxx/
        """
        user_input = user_input.strip()
        match = re.search(r"(BV[0-9A-Za-z]{10,})", user_input)

        if not match:
            raise ValueError("未识别到有效的 BV 号，请检查输入内容。")

        return match.group(1)

    def get_video_aid(self, bvid: str) -> int:
        """
        根据 BV 号获取视频 aid。

        B站评论接口需要 oid 参数。
        对于视频评论区来说，oid 通常就是 aid。
        """
        params = {
            "bvid": bvid
        }

        response = requests.get(
            self.video_info_url,
            headers=self.headers,
            params=params,
            timeout=10
        )
        response.raise_for_status()

        result = response.json()

        if result.get("code") != 0:
            raise ValueError(f"获取视频信息失败：{result.get('message')}")

        aid = result["data"]["aid"]
        title = result["data"].get("title", "未知标题")

        print(f"视频标题：{title}")
        print(f"视频 aid：{aid}")

        return aid

    def fetch_comments(self, content_id: str, max_pages: int = 10):
        """
        实现 BaseCommentCrawler 中要求的抽象方法。

        说明：
        当前项目的完整采集逻辑由 main.py 控制。
        main.py 会一页一页调用 fetch_main_page() 或 fetch_old_page()，
        并且边爬边清洗，判断有效评论数量是否达到目标。

        这个 fetch_comments() 方法主要用于：
        1. 满足 BaseCommentCrawler 基类接口要求
        2. 避免出现“抽象类不能实例化”的报错
        3. 也可以作为简单测试方法使用
        """
        bvid = self.extract_bvid(content_id)
        aid = self.get_video_aid(bvid)

        all_comments = []
        next_page = 0

        for page in range(1, max_pages + 1):
            page_comments, next_page, is_end = self.fetch_main_page(
                bvid=bvid,
                aid=aid,
                next_page=next_page,
                crawl_mode="latest"
            )

            if not page_comments and page == 1:
                page_comments, is_end = self.fetch_old_page(
                    bvid=bvid,
                    aid=aid,
                    page=page,
                    crawl_mode="latest"
                )

            all_comments.extend(page_comments)

            if is_end:
                break

        return all_comments

    def fetch_main_page(
        self,
        bvid: str,
        aid: int,
        next_page: int,
        crawl_mode: str
    ):
        """
        使用新版接口爬取一页评论。

        参数：
        bvid：视频 BV 号
        aid：视频 aid
        next_page：新版接口的下一页游标
        crawl_mode：latest 表示最新评论，hot 表示最热评论

        返回：
        comments：当前页评论列表
        new_next_page：下一页游标
        is_end：是否到达最后一页
        """
        if crawl_mode == "hot":
            mode = 3
        else:
            mode = 2

        params = {
            "type": 1,
            "oid": aid,
            "mode": mode,
            "next": next_page,
            "ps": self.page_size
        }

        response = requests.get(
            self.comment_main_url,
            headers=self.headers,
            params=params,
            timeout=10
        )
        response.raise_for_status()

        result = response.json()

        if result.get("code") != 0:
            print(f"新版接口请求失败：{result.get('message')}")
            return [], next_page, True

        data = result.get("data")
        if not data:
            return [], next_page, True

        replies = data.get("replies")
        if not replies:
            return [], next_page, True

        comments = []
        for raw_comment in replies:
            comment = self.normalize_comment(raw_comment)
            comment["content_id"] = bvid
            comment["aid"] = aid
            comment["crawl_mode"] = crawl_mode
            comments.append(comment)

        cursor = data.get("cursor", {})
        new_next_page = cursor.get("next", 0)
        is_end = cursor.get("is_end", False)

        return comments, new_next_page, is_end

    def fetch_old_page(
        self,
        bvid: str,
        aid: int,
        page: int,
        crawl_mode: str
    ):
        """
        使用旧版接口爬取一页评论。

        当新版接口无法获取评论时，main.py 会切换到这个接口作为备用。
        """
        if crawl_mode == "hot":
            sort = 2
        else:
            sort = 0

        params = {
            "type": 1,
            "oid": aid,
            "pn": page,
            "ps": self.page_size,
            "sort": sort,
            "nohot": 1
        }

        response = requests.get(
            self.comment_old_url,
            headers=self.headers,
            params=params,
            timeout=10
        )
        response.raise_for_status()

        result = response.json()

        if result.get("code") != 0:
            print(f"旧版接口请求失败：{result.get('message')}")
            return [], True

        data = result.get("data")
        if not data:
            return [], True

        replies = data.get("replies")
        if not replies:
            return [], True

        comments = []
        for raw_comment in replies:
            comment = self.normalize_comment(raw_comment)
            comment["content_id"] = bvid
            comment["aid"] = aid
            comment["crawl_mode"] = crawl_mode
            comments.append(comment)

        return comments, False

    def normalize_comment(self, raw_comment: dict):
        """
        将 B站原始评论字段转换成统一评论格式。

        统一字段的好处：
        后续如果接入小红书、抖音、微博等平台，
        只需要把其他平台的原始评论也转换成下面这套字段，
        后面的清洗、情感分析、可视化模块就可以复用。
        """
        member = raw_comment.get("member", {})
        content = raw_comment.get("content", {})

        timestamp = raw_comment.get("ctime", 0)

        if timestamp:
            comment_time = datetime.fromtimestamp(timestamp).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        else:
            comment_time = ""

        return {
            "platform": self.platform,
            "content_id": "",
            "aid": "",
            "crawl_mode": "",
            "user_name": member.get("uname", "未知用户"),
            "user_id": member.get("mid", ""),
            "comment_text": content.get("message", ""),
            "like_count": raw_comment.get("like", 0),
            "reply_count": raw_comment.get("rcount", 0),
            "comment_time": comment_time,
            "rpid": raw_comment.get("rpid", "")
        }

    def save_comments_to_csv(self, comments, save_path: str):
        """
        将原始评论保存到 CSV 文件。

        如果 raw_comments.csv 正在被 Excel 或 WPS 打开，
        程序会自动另存为带时间戳的新文件，避免 PermissionError。
        """
        if not comments:
            print("没有原始评论数据可保存。")
            return ""

        folder = os.path.dirname(save_path)
        if folder and not os.path.exists(folder):
            os.makedirs(folder)

        df = pd.DataFrame(comments)

        try:
            df.to_csv(save_path, index=False, encoding="utf-8-sig")
            final_path = save_path
        except PermissionError:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            name, ext = os.path.splitext(save_path)
            final_path = f"{name}_{timestamp}{ext}"
            df.to_csv(final_path, index=False, encoding="utf-8-sig")

            print("raw_comments.csv 可能正在被 Excel/WPS 占用。")
            print(f"已自动另存为：{final_path}")

        print(f"原始评论数据已保存到：{final_path}")
        print(f"原始评论数量：{len(df)} 条")

        return final_path