import os
import sys
import traceback

from main import main as crawl_and_clean_main
from sentiment_analyzer import SentimentAnalyzer
from visualizer import CommentVisualizer
from report_generator import ReportGenerator


def print_system_header():
    """
    打印系统欢迎信息。
    """
    print("\n========== B站评论舆情分析系统 ==========")
    print("欢迎使用 B站评论舆情分析系统！")
    print("本系统将自动完成：评论采集、情感分析、AI复核、图表生成和报告输出。")
    print("提示：运行前请关闭已打开的 CSV 文件，避免保存失败。")
    print()


def print_module_title(module_number: int, title: str, description: str = ""):
    """
    打印模块标题，让不同阶段之间清晰但不臃肿。
    """
    print()
    print("-" * 60)
    print(f"[{module_number}/4] {title}")
    print("-" * 60)

    if description:
        print(f">>> {description}")


def check_file_exists(file_path: str, module_name: str) -> bool:
    """
    检查某个模块运行后是否生成目标文件。
    """
    if os.path.exists(file_path):
        return True

    print(f"提示：{module_name} 运行后未找到 {file_path}。")
    return False


def run_sentiment_analysis():
    """
    运行情感分析模块。
    """
    clean_file = os.path.join("data", "clean_comments.csv")
    sentiment_file = os.path.join("data", "sentiment_comments.csv")
    summary_file = os.path.join("output", "sentiment_summary.txt")

    analyzer = SentimentAnalyzer()
    analyzer.run(clean_file, sentiment_file, summary_file)


def run_visualization():
    """
    运行可视化模块。
    """
    sentiment_file = os.path.join("data", "sentiment_comments.csv")
    output_folder = "output"

    visualizer = CommentVisualizer()
    visualizer.run(sentiment_file, output_folder)


def run_report_generation():
    """
    运行最终报告生成模块。
    """
    generator = ReportGenerator()
    generator.run()


def print_output_summary():
    """
    打印本轮分析输出文件。
    """
    print()
    print("本轮分析完成！")
    print("核心结果已保存至 data 和 output 文件夹。")

    print("\n核心数据文件：")
    print("1. data/raw_comments.csv")
    print("2. data/clean_comments.csv")
    print("3. data/sentiment_comments.csv")

    print("\n核心输出文件：")
    print("1. output/sentiment_summary.txt")
    print("2. output/visualization_summary.txt")
    print("3. output/final_analysis_report.txt")

    print("\n主分析图表：")
    print("1. output/sentiment_type_bar.png")
    print("2. output/sentiment_score_hist.png")
    print("3. output/sentiment_trend.png")
    print("4. output/influence_comments_bar.png")

    print("\n补充展示图：")
    print("1. output/sentiment_pie.png")
    print("2. output/positive_wordcloud.png")
    print("3. output/negative_wordcloud.png")


def ask_continue() -> bool:
    """
    询问用户是否继续分析其他 B站 视频。
    """
    print()

    while True:
        choice = input("是否继续分析其他 B站 视频？请输入 y/n：").strip().lower()

        if choice in ["y", "yes", "是", "继续"]:
            print("\n即将开始新一轮视频评论舆情分析。")
            return True

        if choice in ["n", "no", "否", "不", "不用", "退出"]:
            print("\n感谢使用 B站评论舆情分析系统，欢迎下次继续使用！")
            return False

        print("输入无效，请输入 y 表示继续，或输入 n 表示退出。")


def run_one_analysis_round(round_number: int) -> bool:
    """
    执行一轮完整分析流程。
    """
    if round_number > 1:
        print()
        print(f"========== 第 {round_number} 轮视频评论舆情分析 ==========")

    # [1/4] 在 main.py 中用户输入完参数后显示
    crawl_and_clean_main()

    if not check_file_exists(
        os.path.join("data", "clean_comments.csv"),
        "评论采集与清洗"
    ):
        print("流程终止：缺少 clean_comments.csv，无法继续情感分析。")
        return False

    print_module_title(
        2,
        "情感分析与 AI 复核",
        "正在进行 SnowNLP 情感初判、规则修正和 Gemini AI 关键评论复核。"
    )

    run_sentiment_analysis()

    if not check_file_exists(
        os.path.join("data", "sentiment_comments.csv"),
        "情感分析与 AI 复核"
    ):
        print("流程终止：缺少 sentiment_comments.csv，无法继续可视化。")
        return False

    print_module_title(
        3,
        "生成舆情可视化图表",
        "正在生成情感分布、得分分布、时间趋势、高影响力评论排行和补充词云图。"
    )

    run_visualization()

    expected_files = [
        os.path.join("output", "sentiment_type_bar.png"),
        os.path.join("output", "sentiment_pie.png"),
        os.path.join("output", "sentiment_score_hist.png"),
        os.path.join("output", "sentiment_trend.png"),
        os.path.join("output", "influence_comments_bar.png"),
        os.path.join("output", "positive_wordcloud.png"),
        os.path.join("output", "negative_wordcloud.png"),
        os.path.join("output", "visualization_summary.txt")
    ]

    missing_files = [
        file_path for file_path in expected_files
        if not os.path.exists(file_path)
    ]

    if missing_files:
        print("\n提示：以下可视化文件未生成，可能是对应数据不足或文件被占用：")
        for file_path in missing_files:
            print(f"- {file_path}")
        print("系统将继续生成最终报告。")

    print_module_title(
        4,
        "生成舆情分析报告",
        "正在根据统计结果、AI复核结果和代表性评论生成最终分析报告。"
    )

    run_report_generation()

    if not check_file_exists(
        os.path.join("output", "final_analysis_report.txt"),
        "生成舆情分析报告"
    ):
        print("流程结束，但最终报告未成功生成。")
        return False

    print_output_summary()
    return True


def main():
    """
    一键运行入口。

    支持循环分析多个 B站 视频：
    1. 每轮分析一个视频
    2. 每轮结束后询问是否继续
    3. 用户选择退出时显示感谢语
    """
    print_system_header()

    round_number = 1

    while True:
        try:
            success = run_one_analysis_round(round_number)

            if not success:
                print("\n本轮分析未完整完成，请根据上方提示检查问题后重新运行。")
                break

            should_continue = ask_continue()

            if not should_continue:
                break

            round_number += 1

        except KeyboardInterrupt:
            print("\n\n检测到用户手动中断程序。")
            print("感谢使用 B站评论舆情分析系统，程序已安全退出。")
            sys.exit(0)

        except Exception as error:
            print("\n程序运行过程中发生错误：")
            print(error)
            print("\n详细错误信息如下：")
            traceback.print_exc()
            print("\n请根据报错信息检查对应模块。")
            break


if __name__ == "__main__":
    main()