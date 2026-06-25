import math
import os
import time

import pandas as pd

from bilibili_crawler import BilibiliCommentCrawler
from cleaner import CommentCleaner


# 系统保护上限：防止程序无限爬取
MAX_RAW_COMMENTS = 500

# 用户单次最多可以要求的有效评论数量
MAX_VALID_TARGET = 300

# B站评论接口每页通常返回 20 条左右
PAGE_SIZE = 20


def get_valid_target_count() -> int:
    """
    获取用户输入的目标有效评论数量。

    有效评论数量指：
    清洗之后 clean_comments.csv 中真正用于分析的评论数量。
    """
    user_input = input(
        "请输入目标有效评论数量，建议 50-300，最大 300："
    ).strip()

    if not user_input.isdigit():
        print("输入无效，默认目标有效评论数量为 100 条。")
        return 100

    count = int(user_input)

    if count <= 0:
        print("目标数量必须大于 0，默认设置为 100 条。")
        return 100

    if count > MAX_VALID_TARGET:
        print(
            f"目标有效评论数量不能超过 {MAX_VALID_TARGET} 条，"
            f"已自动调整为 {MAX_VALID_TARGET} 条。"
        )
        return MAX_VALID_TARGET

    return count


def get_user_crawl_mode() -> str:
    """
    获取评论采集模式。
    """
    print("请选择评论采集模式：")
    print("1. 最新评论：适合观察情绪随时间变化")
    print("2. 最热评论：适合分析高赞观点和舆情焦点")

    choice = input("请输入选项 1 或 2：").strip()

    if choice == "2":
        return "hot"

    return "latest"


def build_clean_dataframe(raw_comments, cleaner: CommentCleaner) -> pd.DataFrame:
    """
    将当前已爬取的原始评论临时清洗成有效评论 DataFrame。

    这个函数用于“边爬边判断有效评论数量”。
    """
    if not raw_comments:
        return pd.DataFrame()

    df = pd.DataFrame(raw_comments)

    if "comment_text" not in df.columns:
        return pd.DataFrame()

    # 删除空评论
    df = df.dropna(subset=["comment_text"]).copy()

    # 优先按 rpid 去重，因为 rpid 是 B站评论的唯一 ID
    if "rpid" in df.columns:
        df = df.drop_duplicates(subset=["rpid"])
    else:
        df = df.drop_duplicates(subset=["comment_text"])

    # 保留原始评论
    df["original_comment_text"] = df["comment_text"]

    # 清洗文本
    df["clean_text"] = df["comment_text"].apply(cleaner.clean_text)

    # 判断是否为低信息评论
    df["is_low_value"] = df["clean_text"].apply(
        cleaner.is_low_value_comment
    )

    # 只保留有效评论
    df_clean = df[df["is_low_value"] == False].copy()

    # 删除辅助列
    df_clean = df_clean.drop(columns=["is_low_value"])

    return df_clean


def save_clean_dataframe(df_clean: pd.DataFrame, save_path: str) -> str:
    """
    保存清洗后的有效评论。
    """
    if df_clean.empty:
        print("没有有效评论数据可保存。")
        return ""

    folder = os.path.dirname(save_path)

    if folder and not os.path.exists(folder):
        os.makedirs(folder)

    try:
        df_clean.to_csv(save_path, index=False, encoding="utf-8-sig")
        final_path = save_path
    except PermissionError:
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name, ext = os.path.splitext(save_path)
        final_path = f"{name}_{timestamp}{ext}"

        df_clean.to_csv(final_path, index=False, encoding="utf-8-sig")

        print("clean_comments.csv 可能正在被 Excel/WPS 占用。")
        print(f"已自动另存为：{final_path}")

    print(f"清洗后有效评论数据已保存到：{final_path}")
    print(f"保存的有效评论数量：{len(df_clean)} 条")

    return final_path


def preview_comments(raw_comments, df_clean):
    """
    打印部分原始评论和有效评论预览。
    """
    print()
    print("-" * 60)
    print("原始评论前 3 条预览")
    print("-" * 60)

    for index, item in enumerate(raw_comments[:3], start=1):
        print()
        print(f"【原始评论 {index}】")
        print(f"用户：{item.get('user_name')}")
        print(f"评论：{item.get('comment_text')}")
        print(f"点赞数：{item.get('like_count')}")
        print(f"时间：{item.get('comment_time')}")
        print(f"评论ID：{item.get('rpid')}")
        print("-" * 60)

    print()
    print("-" * 60)
    print("清洗后有效评论前 3 条预览")
    print("-" * 60)

    for index, (_, row) in enumerate(df_clean.head(3).iterrows(), start=1):
        print()
        print(f"【清洗结果 {index}】")
        print(f"原评论：{row['original_comment_text']}")
        print()
        print(f"清洗后：{row['clean_text']}")
        print("-" * 60)


def main():
    """
    程序主入口。

    最终采集策略：
    1. 用户输入视频链接或 BV 号
    2. 用户输入目标有效评论数量
    3. 用户选择最新评论或最热评论
    4. 程序最多爬取 500 条原始评论
    5. 每爬一页就临时清洗并统计有效评论数
    6. 有效评论数量达到或超过目标后停止
    7. 如果清洗后有效评论超过目标数量，保存 clean_comments.csv 时只保留前 N 条
    8. raw_comments.csv 保留全部原始评论，便于追溯
    """

    user_input = input("请输入 B站 视频链接或 BV号：").strip()
    target_valid_count = get_valid_target_count()
    crawl_mode = get_user_crawl_mode()

    print()
    print("-" * 60)
    print("[1/4] 评论采集与清洗")
    print("-" * 60)

    crawler = BilibiliCommentCrawler()
    cleaner = CommentCleaner()

    bvid = crawler.extract_bvid(user_input)
    print(f"识别到 BV 号：{bvid}")

    aid = crawler.get_video_aid(bvid)

    max_pages = math.ceil(MAX_RAW_COMMENTS / PAGE_SIZE)

    raw_comments = []
    df_clean = pd.DataFrame()

    next_page = 0
    use_old_api = False

    print(f"目标有效评论数量：{target_valid_count} 条")
    print(f"单次最大原始评论爬取上限：{MAX_RAW_COMMENTS} 条")
    print(f"采集模式：{crawl_mode}")

    for page in range(1, max_pages + 1):
        if len(raw_comments) >= MAX_RAW_COMMENTS:
            print("已达到最大原始评论爬取上限，停止爬取。")
            break

        print(f"\n正在爬取第 {page} 页评论……")

        try:
            if not use_old_api:
                page_comments, next_page, is_end = crawler.fetch_main_page(
                    bvid=bvid,
                    aid=aid,
                    next_page=next_page,
                    crawl_mode=crawl_mode
                )

                # 如果新版接口第一页没有评论，则切换旧版接口
                if not page_comments and page == 1:
                    print("新版接口未获取到评论，切换为旧版接口。")
                    use_old_api = True

                    page_comments, is_end = crawler.fetch_old_page(
                        bvid=bvid,
                        aid=aid,
                        page=page,
                        crawl_mode=crawl_mode
                    )

            else:
                page_comments, is_end = crawler.fetch_old_page(
                    bvid=bvid,
                    aid=aid,
                    page=page,
                    crawl_mode=crawl_mode
                )

            if not page_comments:
                print("当前页未获取到评论，停止爬取。")
                break

            remain_count = MAX_RAW_COMMENTS - len(raw_comments)
            raw_comments.extend(page_comments[:remain_count])

            # 每爬一页，就临时清洗并统计有效评论数量
            df_clean = build_clean_dataframe(raw_comments, cleaner)

            print(
                f"当前原始评论数量：{len(raw_comments)} 条，"
                f"当前有效评论数量：{len(df_clean)} 条"
            )

            if len(df_clean) >= target_valid_count:
                print("已达到目标有效评论数量，停止爬取。")
                break

            if is_end:
                print("已经到达评论最后一页，停止爬取。")
                break

            time.sleep(1)

        except Exception as error:
            print(f"爬取第 {page} 页时发生错误：{error}")
            break

    print("\n评论采集结束")
    print(f"最终原始评论数量：{len(raw_comments)} 条")
    print(f"清洗后实际有效评论数量：{len(df_clean)} 条")

    if len(df_clean) < target_valid_count:
        print(
            f"提示：未达到目标有效评论数量 {target_valid_count} 条。"
            "可能原因是视频评论较少或低信息评论较多。"
        )
        print("建议：降低目标数量、换一个评论更多的视频，或切换最新/最热模式。")

    elif len(df_clean) > target_valid_count:
        print(
            f"当前有效评论数量为 {len(df_clean)} 条，"
            f"超过目标数量 {target_valid_count} 条。"
        )
        print(f"保存 clean_comments.csv 时将截取前 {target_valid_count} 条有效评论。")
        df_clean = df_clean.head(target_valid_count).copy()

    else:
        print(f"已刚好达到目标有效评论数量：{target_valid_count} 条。")

    raw_save_path = os.path.join("data", "raw_comments.csv")
    clean_save_path = os.path.join("data", "clean_comments.csv")

    # raw_comments.csv 保存全部原始评论，不截断
    crawler.save_comments_to_csv(raw_comments, raw_save_path)

    # clean_comments.csv 只保存目标数量的有效评论
    save_clean_dataframe(df_clean, clean_save_path)

    if raw_comments and not df_clean.empty:
        preview_comments(raw_comments, df_clean)

    print("\n评论采集与清洗完成。")
    print("已生成：data/raw_comments.csv")
    print("已生成：data/clean_comments.csv")


if __name__ == "__main__":
    main()