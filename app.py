def render_home():
    st.markdown("""
    # 📊 BiliScope · 舆情智能分析平台

    ## 🎯 面向用户
    - UP主：优化内容
    - MCN机构：账号运营
    - 品牌方：舆情监控

    ## ⚡ 核心能力
    - 视频舆情分析
    - 评论情绪识别
    - 风险预警系统
    - 多视频对比
    """)
    
import html
import math
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# =========================
# 路径设置
# =========================
BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"

sys.path.append(str(SRC_DIR))

from bilibili_crawler import BilibiliCommentCrawler
from cleaner import CommentCleaner
from sentiment_analyzer import SentimentAnalyzer
from visualizer import CommentVisualizer
from report_generator import ReportGenerator


# =========================
# 基础参数
# =========================
MAX_RAW_COMMENTS = 500
PAGE_SIZE = 20


# =========================
# 页面配置
# =========================
st.set_page_config(
    page_title="B站评论舆情分析系统",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================
# 自定义 CSS
# =========================
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #F4F7FB 0%, #EEF3FA 100%);
    }

    .hero {
        padding: 34px 38px;
        border-radius: 26px;
        background:
            radial-gradient(circle at top right, rgba(56,189,248,0.45), transparent 28%),
            linear-gradient(135deg, #0F172A 0%, #1E3A8A 55%, #0369A1 100%);
        color: white;
        box-shadow: 0 20px 50px rgba(15, 23, 42, 0.25);
        margin-bottom: 24px;
    }

    .hero-title {
        font-size: 38px;
        font-weight: 900;
        margin-bottom: 8px;
        letter-spacing: 0.5px;
    }

    .hero-en {
        font-size: 15px;
        letter-spacing: 1.2px;
        opacity: 0.75;
        margin-bottom: 10px;
        text-transform: uppercase;
    }

    .hero-subtitle {
        font-size: 16px;
        opacity: 0.92;
        line-height: 1.8;
        margin-bottom: 16px;
    }

    .tag {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 999px;
        background: rgba(255,255,255,0.15);
        border: 1px solid rgba(255,255,255,0.25);
        margin-right: 8px;
        margin-bottom: 6px;
        font-size: 13px;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #111827 0%, #1E293B 100%) !important;
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    section[data-testid="stSidebar"] * {
        color: #F8FAFC;
    }

    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span {
        color: #F8FAFC !important;
    }

    section[data-testid="stSidebar"] input {
        color: #F8FAFC !important;
        background-color: #0F172A !important;
        border: 1px solid #334155 !important;
    }

    section[data-testid="stSidebar"] input::placeholder {
        color: #CBD5E1 !important;
    }

    .sidebar-title {
        font-size: 24px;
        font-weight: 900;
        color: #FFFFFF;
        margin-bottom: 10px;
    }

    .sidebar-desc {
        color: #CBD5E1;
        font-size: 13px;
        line-height: 1.7;
        margin-bottom: 20px;
    }

    .section-title {
        font-size: 22px;
        font-weight: 900;
        color: #0F172A;
        margin-bottom: 6px;
    }

    .section-desc {
        font-size: 14px;
        color: #64748B;
        margin-bottom: 16px;
        line-height: 1.7;
    }

    .status-panel {
        background:
            linear-gradient(135deg, rgba(255,255,255,0.98), rgba(248,250,252,0.98));
        border: 1px solid #E2E8F0;
        border-radius: 22px;
        padding: 24px 26px;
        box-shadow: 0 14px 34px rgba(15, 23, 42, 0.08);
        margin-bottom: 18px;
    }

    .status-main {
        display: grid;
        grid-template-columns: 1.35fr 0.65fr 1.25fr;
        gap: 24px;
        align-items: start;
    }

   .status-block {
        min-height: 0;
    }

    .status-label {
        font-size: 13px;
        color: #64748B;
        font-weight: 800;
        margin-bottom: 7px;
        line-height: 16px;
    }

    .status-value {
        font-size: 26px;
        color: #0F172A;
        font-weight: 900;
        line-height: 36px;
        height: 36px;
        white-space: nowrap;
    }

    .status-progress {
        font-size: 32px;
        color: #16A34A;
        font-weight: 900;
        line-height: 36px;
        height: 36px;
        white-space: nowrap;
    }

    .status-time {
        font-size: 24px;
        color: #0F172A;
        font-weight: 900;
        line-height: 36px;
        height: 36px;
        white-space: nowrap;
    }

    .status-sub {
        font-size: 14px;
        color: #475569;
        line-height: 20px;
        margin-top: 10px;
        margin-bottom: 8px;
        white-space: nowrap;
    }

    .stage-row {
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
        margin-top: 8px;
    }

    .stage-chip {
        border-radius: 999px;
        padding: 7px 12px;
        font-size: 13px;
        font-weight: 800;
        border: 1px solid #CBD5E1;
        background: #F8FAFC;
        color: #475569;
    }

    .stage-done {
        background: #ECFDF5;
        color: #047857;
        border-color: #A7F3D0;
    }

    .stage-active {
        background: #DBEAFE;
        color: #1D4ED8;
        border-color: #93C5FD;
    }

    .stage-pending {
        background: #F8FAFC;
        color: #64748B;
    }

    .log-box {
        background: #F3F4F6;
        border: 1px solid #CBD5E1;
        border-radius: 14px;
        padding: 12px 14px;
        height: 210px;
        overflow-y: scroll;
        color: #111827;
        font-size: 14px;
        line-height: 1.85;
        scrollbar-width: auto;
        scrollbar-color: #4B5563 #D1D5DB;
    }

    .log-box::-webkit-scrollbar {
        width: 14px;
        display: block;
    }

    .log-box::-webkit-scrollbar-track {
        background: #D1D5DB;
        border-radius: 999px;
    }

    .log-box::-webkit-scrollbar-thumb {
        background: #4B5563;
        border-radius: 999px;
        border: 3px solid #D1D5DB;
    }

    .log-line {
        color: #111827;
        margin-bottom: 7px;
        word-break: break-all;
    }

    .metric-card {
        background: white;
        padding: 20px 22px;
        border-radius: 20px;
        box-shadow: 0 12px 28px rgba(15, 23, 42, 0.08);
        border: 1px solid rgba(226, 232, 240, 0.95);
        height: 132px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        position: relative;
        overflow: hidden;
    }

    .metric-card::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #2563EB, #38BDF8);
    }

    .metric-label {
        color: #64748B;
        font-size: 14px;
        margin-bottom: 10px;
        font-weight: 700;
    }

    .metric-value {
        color: #0F172A;
        font-size: 30px;
        font-weight: 900;
        line-height: 1.1;
    }

    .metric-note {
        color: #94A3B8;
        font-size: 12px;
        margin-top: 10px;
        line-height: 1.4;
    }

    .overview-text {
        font-size: 15px;
        line-height: 1.9;
        color: #334155;
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 18px 20px;
        margin-bottom: 16px;
    }

    button[data-baseweb="tab"] p {
        color: #334155 !important;
        font-weight: 700 !important;
    }

    button[data-baseweb="tab"][aria-selected="true"] p {
        color: #EF4444 !important;
        font-weight: 900 !important;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        border-bottom-color: #EF4444 !important;
    }

    .stImage figcaption {
        color: #475569 !important;
        font-size: 13px !important;
        text-align: center !important;
    }

    div[data-testid="stDataFrame"] {
        border-radius: 16px;
        overflow: hidden;
    }

    .hi-comment-card {
        background: #FFFFFF !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 16px !important;
        padding: 16px 18px !important;
        margin-bottom: 12px !important;
        box-shadow: 0 8px 20px rgba(15, 23, 42, 0.06) !important;
    }

    .hi-comment-title {
        color: #1E293B !important;
        font-size: 15px !important;
        font-weight: 900 !important;
        margin-bottom: 8px !important;
    }

    .hi-comment-text {
        color: #0F172A !important;
        font-size: 15px !important;
        line-height: 1.8 !important;
        font-weight: 600 !important;
        margin-bottom: 12px !important;
        word-break: break-word !important;
    }

    .hi-comment-tags {
        display: flex !important;
        gap: 8px !important;
        flex-wrap: wrap !important;
    }

    .hi-tag-like {
        display: inline-block !important;
        background: #EFF6FF !important;
        border: 1px solid #BFDBFE !important;
        color: #1E3A8A !important;
        border-radius: 999px !important;
        padding: 5px 10px !important;
        font-size: 12px !important;
        font-weight: 800 !important;
    }

    .hi-tag-score {
        display: inline-block !important;
        background: #F0FDF4 !important;
        border: 1px solid #BBF7D0 !important;
        color: #166534 !important;
        border-radius: 999px !important;
        padding: 5px 10px !important;
        font-size: 12px !important;
        font-weight: 800 !important;
    }

    .footer {
    text-align: center;
    color: #94A3B8;
    font-size: 13px;
    padding: 22px 0 8px 0;
}

    /* ========== 进度日志折叠栏：白天/夜间模式都清楚 ========== */
    div[data-testid="stExpander"] {
        background: #F8FAFC !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 12px !important;
        margin-top: 10px !important;
        margin-bottom: 14px !important;
        overflow: hidden !important;
    }

    /* 折叠栏最外层 */
    div[data-testid="stExpander"] details {
        background: #F8FAFC !important;
        color: #0F172A !important;
    }

    /* 折叠栏标题区域：重点改这里 */
    div[data-testid="stExpander"] details summary {
        background: #F3F4F6 !important;
        color: #0F172A !important;
        font-weight: 800 !important;
        padding: 12px 16px !important;
        border-radius: 12px 12px 0 0 !important;
    }

    /* 标题里面所有文字、图标、箭头 */
    div[data-testid="stExpander"] details summary * {
        color: #0F172A !important;
        fill: #0F172A !important;
        stroke: #0F172A !important;
    }

    /* 有些版本 Streamlit 会把标题内容套在 div 里，也强制改 */
    div[data-testid="stExpander"] div {
        color: #0F172A !important;
    }

    /* 展开后的内容区域 */
    div[data-testid="stExpanderDetails"] {
        background: #FFFFFF !important;
        color: #111827 !important;
        padding: 12px 16px !important;
    }

    /* 日志框 */
    .log-box {
        background: #FFFFFF !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 12px !important;
        padding: 12px 14px !important;
        height: 210px !important;
        overflow-y: scroll !important;
        color: #111827 !important;
        font-size: 14px !important;
        line-height: 1.85 !important;
    }

    /* 日志文字 */
    .log-line {
        color: #111827 !important;
        margin-bottom: 7px !important;
        word-break: break-all !important;
    }

    /* ========== 自定义完成提示 ========== */
    .success-box {
        background: #DCFCE7 !important;
        border: 1px solid #86EFAC !important;
        color: #166534 !important;
        border-radius: 12px !important;
        padding: 14px 18px !important;
        font-size: 16px !important;
        font-weight: 850 !important;
        margin-top: 12px !important;
        margin-bottom: 18px !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================
# 工具函数
# =========================
def ensure_dirs():
    DATA_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)


def build_clean_dataframe(raw_comments, cleaner: CommentCleaner) -> pd.DataFrame:
    if not raw_comments:
        return pd.DataFrame()

    df = pd.DataFrame(raw_comments)

    if "comment_text" not in df.columns:
        return pd.DataFrame()

    df = df.dropna(subset=["comment_text"]).copy()

    if "rpid" in df.columns:
        df = df.drop_duplicates(subset=["rpid"])
    else:
        df = df.drop_duplicates(subset=["comment_text"])

    df["original_comment_text"] = df["comment_text"]
    df["clean_text"] = df["comment_text"].apply(cleaner.clean_text)
    df["is_low_value"] = df["clean_text"].apply(cleaner.is_low_value_comment)

    df_clean = df[df["is_low_value"] == False].copy()
    df_clean = df_clean.drop(columns=["is_low_value"])

    return df_clean


def save_clean_dataframe(df_clean: pd.DataFrame, save_path: Path):
    if df_clean.empty:
        return ""

    df_clean.to_csv(save_path, index=False, encoding="utf-8-sig")
    return str(save_path)


def save_raw_comments(crawler, raw_comments, save_path: Path):
    crawler.save_comments_to_csv(raw_comments, str(save_path))


def load_sentiment_result():
    sentiment_path = DATA_DIR / "sentiment_comments.csv"

    if not sentiment_path.exists():
        return None

    return pd.read_csv(sentiment_path, encoding="utf-8-sig")


def load_report_text():
    report_path = OUTPUT_DIR / "final_analysis_report.txt"

    if not report_path.exists():
        return ""

    with open(report_path, "r", encoding="utf-8") as file:
        return file.read()


def get_ai_review_count(df: pd.DataFrame) -> int:
    if "ai_reviewed" not in df.columns:
        return 0

    return int(df["ai_reviewed"].astype(str).str.lower().eq("true").sum())


def get_basic_metrics(df: pd.DataFrame):
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
    dispute_index = round(1 - abs(positive_ratio - negative_ratio) / 100, 4)
    ai_review_count = get_ai_review_count(df)

    return {
        "total": total,
        "positive_count": positive_count,
        "neutral_count": neutral_count,
        "negative_count": negative_count,
        "positive_ratio": positive_ratio,
        "neutral_ratio": neutral_ratio,
        "negative_ratio": negative_ratio,
        "avg_score": avg_score,
        "dispute_index": dispute_index,
        "ai_review_count": ai_review_count,
    }


def build_overview_paragraph(metrics: dict) -> str:
    positive_ratio = metrics["positive_ratio"]
    negative_ratio = metrics["negative_ratio"]
    neutral_ratio = metrics["neutral_ratio"]
    avg_score = metrics["avg_score"]
    dispute_index = metrics["dispute_index"]
    total = metrics["total"]

    if positive_ratio >= 60:
        main_tone = "整体来看，评论区以正向情绪为主，说明多数用户对视频内容或相关讨论对象表现出认可、支持或共鸣。"
    elif negative_ratio >= 45:
        main_tone = "整体来看，评论区负向情绪较为突出，说明用户中存在较明显的质疑、不满或焦虑表达。"
    elif positive_ratio > negative_ratio:
        main_tone = "整体来看，评论区略偏正向，但负向评论仍占有一定比例，说明舆情并非完全一边倒。"
    else:
        main_tone = "整体来看，评论区正负态度较为接近，用户观点存在一定分化。"

    if dispute_index >= 0.85:
        dispute_text = "争议度指数较高，说明评论区正负观点比较接近，讨论分歧较明显。"
    elif dispute_index >= 0.65:
        dispute_text = "争议度指数处于中等水平，说明评论区存在一定争议，但仍能看出主要情绪方向。"
    else:
        dispute_text = "争议度指数较低，说明评论区观点相对集中，主流态度较明确。"

    return (
        f"本次共分析 {total} 条有效评论，平均情感得分为 {avg_score}。"
        f"其中正向评论占比 {positive_ratio}%，中立评论占比 {neutral_ratio}%，"
        f"负向评论占比 {negative_ratio}%。{main_tone}{dispute_text}"
        "因此，本次评论区既可以通过情感分布观察整体态度，也需要结合高影响力评论进一步判断舆情风险和讨论焦点。"
    )


def safe_download_bytes(text: str) -> bytes:
    return text.encode("utf-8-sig")


def estimate_remaining_seconds(current_count: int, target_count: int, start_time: float):
    if current_count <= 0:
        return None

    elapsed = time.time() - start_time
    speed = current_count / elapsed if elapsed > 0 else 0

    if speed <= 0:
        return None

    remaining_count = max(target_count - current_count, 0)
    return int(remaining_count / speed)


def get_remaining_text(remain_seconds):
    if remain_seconds is None:
        return "预计剩余：计算中"
    if remain_seconds <= 0:
        return "预计剩余：即将完成"
    return f"预计剩余：约 {remain_seconds} 秒"


def render_stage_chips(active_stage: int):
    stages = ["评论采集", "情感分析", "图表生成", "报告输出"]
    html_text = '<div class="stage-row">'

    for idx, name in enumerate(stages, start=1):
        if idx < active_stage:
            cls = "stage-chip stage-done"
            label = f"已完成 · {name}"
        elif idx == active_stage:
            cls = "stage-chip stage-active"
            label = f"进行中 · {name}"
        else:
            cls = "stage-chip stage-pending"
            label = f"待开始 · {name}"

        html_text += f'<span class="{cls}">{label}</span>'

    html_text += "</div>"
    return html_text


def update_status(status_area, log_lines, stage: int, percent: int, action: str, remain_seconds=None):
    safe_action = html.escape(str(action))
    remain_text = get_remaining_text(remain_seconds)
    stage_html = render_stage_chips(stage)

    status_area.markdown(
        f"""<div class="status-panel">
<div class="status-main">
    <div class="status-block">
        <div class="status-label">当前任务</div>
        <div class="status-value">{safe_action}</div>
    </div>
    <div class="status-block">
        <div class="status-label">当前进度</div>
        <div class="status-progress">{percent}%</div>
    </div>
    <div class="status-block">
        <div class="status-label">时间估计</div>
        <div class="status-time">{remain_text}</div>
    </div>
</div>
<div class="status-sub">
    系统正在自动执行评论采集、清洗、情感分析、可视化和报告生成流程。
</div>
{stage_html}
</div>""",
        unsafe_allow_html=True
    )


def add_log(log_lines, message: str):
    log_lines.append(str(message))

    log_area = st.session_state.get("log_area", None)

    if log_area is not None:
        safe_html = ""

        for line in log_lines[-120:]:
            safe_line = html.escape(line)
            safe_html += f'<div class="log-line">{safe_line}</div>'

        log_area.markdown(
            f"""
            <div class="log-box">
                {safe_html}
            </div>
            """,
            unsafe_allow_html=True
        )


def render_log_box(log_lines):
    safe_html = ""

    for line in log_lines[-120:]:
        safe_line = html.escape(line)
        safe_html += f'<div class="log-line">{safe_line}</div>'

    st.markdown(
        f"""
        <div class="log-box">
            {safe_html}
        </div>
        """,
        unsafe_allow_html=True
    )


def image_with_download(path: Path, caption: str, button_label: str):
    if not path.exists():
        st.warning(f"未找到图表：{path}")
        return

    st.image(str(path), caption=caption, use_container_width=True)

    with open(path, "rb") as file:
        image_bytes = file.read()

    st.download_button(
        label=button_label,
        data=image_bytes,
        file_name=path.name,
        mime="image/png",
        use_container_width=True
    )


CHART_COLORS = {
    "positive": "#4F7CAC",      # 低饱和沉稳蓝
    "neutral": "#A8B3C2",       # 雾灰蓝
    "negative": "#C95F66",      # 柔和珊瑚红
    "primary": "#4F7CAC",
    "primary_light": "rgba(79, 124, 172, 0.16)",
    "grid": "#EEF2F7",
    "axis": "#CBD5E1",
    "text": "#162033",
    "subtext": "#64748B",
    "white": "#FFFFFF"
}


def apply_clean_plotly_layout(fig, title: str, height: int = 430):
    """
    统一高级浅色图表风格。
    """
    fig.update_layout(
        title=dict(
            text=title,
            x=0.02,
            y=0.95,
            xanchor="left",
            yanchor="top",
            font=dict(
                size=20,
                color=CHART_COLORS["text"],
                family="Microsoft YaHei"
            )
        ),
        template="plotly_white",
        height=height,
        paper_bgcolor=CHART_COLORS["white"],
        plot_bgcolor=CHART_COLORS["white"],
        font=dict(
            size=14,
            color=CHART_COLORS["text"],
            family="Microsoft YaHei"
        ),
        margin=dict(l=58, r=36, t=82, b=58),
        hoverlabel=dict(
            bgcolor="#162033",
            font_size=13,
            font_color="#FFFFFF",
            font_family="Microsoft YaHei"
        ),
        legend=dict(
            bgcolor="rgba(255,255,255,0)",
            borderwidth=0,
            font=dict(size=13, color=CHART_COLORS["text"])
        )
    )

    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        linecolor=CHART_COLORS["axis"],
        tickfont=dict(color=CHART_COLORS["subtext"], size=13),
        title_font=dict(color=CHART_COLORS["subtext"], size=14)
    )

    fig.update_yaxes(
        showgrid=True,
        gridcolor=CHART_COLORS["grid"],
        zeroline=False,
        linecolor=CHART_COLORS["axis"],
        tickfont=dict(color=CHART_COLORS["subtext"], size=13),
        title_font=dict(color=CHART_COLORS["subtext"], size=14)
    )

    return fig


def download_png_button(path: Path, button_label: str):
    """
    保留原 PNG 下载功能。
    网页上用 Plotly 动态图，下载时仍下载 visualizer.py 生成的图片。
    """
    if not path.exists():
        st.warning(f"未找到可下载图表：{path}")
        return

    with open(path, "rb") as file:
        image_bytes = file.read()

    st.download_button(
        label=button_label,
        data=image_bytes,
        file_name=path.name,
        mime="image/png",
        use_container_width=True
    )

def render_sentiment_type_plotly(df: pd.DataFrame):
    """
    动态图：情感类别分布柱状图，高级浅色版。
    """
    if "sentiment_type" not in df.columns:
        st.warning("缺少 sentiment_type 字段，无法绘制情感类别分布图。")
        return

    total = len(df)

    plot_df = (
        df["sentiment_type"]
        .value_counts()
        .reindex(["正向", "中立", "负向"], fill_value=0)
        .reset_index()
    )

    plot_df.columns = ["情感类型", "评论数量"]
    plot_df["占比"] = (plot_df["评论数量"] / total * 100).round(2)
    plot_df["标签"] = (
        plot_df["评论数量"].astype(str)
        + "条<br>"
        + plot_df["占比"].astype(str)
        + "%"
    )

    color_map = {
        "正向": CHART_COLORS["positive"],
        "中立": CHART_COLORS["neutral"],
        "负向": CHART_COLORS["negative"]
    }

    fig = go.Figure()

    for _, row in plot_df.iterrows():
        fig.add_trace(
            go.Bar(
                x=[row["情感类型"]],
                y=[row["评论数量"]],
                name=row["情感类型"],
                marker=dict(
                    color=color_map.get(row["情感类型"], "#64748B"),
                    line=dict(width=0),
                    opacity=0.95
                ),
                text=[row["标签"]],
                textposition="outside",
                textfont=dict(size=13, color=CHART_COLORS["text"]),
                customdata=[[row["占比"]]],
                hovertemplate=(
                    "情感类型：%{x}<br>"
                    "评论数量：%{y} 条<br>"
                    "占比：%{customdata[0]}%<extra></extra>"
                )
            )
        )

    fig = apply_clean_plotly_layout(
        fig,
        title="评论情感类别分布",
        height=430
    )

    fig.update_layout(
        showlegend=False,
        bargap=0.45,
        yaxis_title="评论数量",
        xaxis_title=""
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "displayModeBar": False,
            "responsive": True
        }
    )


def render_sentiment_score_hist_plotly(df: pd.DataFrame):
    """
    动态图：情感得分分布直方图，高级浅色版。
    """
    if "sentiment_score" not in df.columns:
        st.warning("缺少 sentiment_score 字段，无法绘制情感得分分布图。")
        return

    plot_df = df.copy()
    plot_df["sentiment_score"] = pd.to_numeric(
        plot_df["sentiment_score"],
        errors="coerce"
    )

    plot_df = plot_df.dropna(subset=["sentiment_score"])

    if plot_df.empty:
        st.warning("没有有效情感得分数据，无法绘制情感得分分布图。")
        return

    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=plot_df["sentiment_score"],
            nbinsx=20,
            marker=dict(
                color=CHART_COLORS["primary"],
                line=dict(color="#FFFFFF", width=1.2)
            ),
            opacity=0.88,
            hovertemplate="情感得分：%{x}<br>评论数量：%{y}<extra></extra>"
        )
    )

    fig = apply_clean_plotly_layout(
        fig,
        title="评论情感得分分布",
        height=430
    )

    fig.update_layout(
        bargap=0.08,
        showlegend=False,
        xaxis_title="情感得分",
        yaxis_title="评论数量"
    )

    fig.update_xaxes(range=[0, 1])

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "displayModeBar": False,
            "responsive": True
        }
    )


def render_sentiment_trend_plotly(df: pd.DataFrame):
    """
    动态图：情绪时间趋势折线图，高级浅色版。
    """
    required_columns = ["comment_time", "sentiment_score"]

    for column in required_columns:
        if column not in df.columns:
            st.warning(f"缺少 {column} 字段，无法绘制情绪时间趋势图。")
            return

    plot_df = df.copy()

    plot_df["comment_time"] = pd.to_datetime(
        plot_df["comment_time"],
        errors="coerce"
    )

    plot_df["sentiment_score"] = pd.to_numeric(
        plot_df["sentiment_score"],
        errors="coerce"
    )

    plot_df = plot_df.dropna(subset=["comment_time", "sentiment_score"])

    if plot_df.empty:
        st.warning("没有有效时间数据，无法绘制情绪时间趋势图。")
        return

    min_time = plot_df["comment_time"].min()
    max_time = plot_df["comment_time"].max()
    time_span = max_time - min_time

    if time_span <= pd.Timedelta(days=2):
        plot_df["time_group"] = plot_df["comment_time"].dt.floor("h")
        x_title = "评论时间（按小时）"
    else:
        plot_df["time_group"] = plot_df["comment_time"].dt.date
        x_title = "评论时间（按日期）"

    trend_df = (
        plot_df.groupby("time_group")
        .agg(
            平均情感得分=("sentiment_score", "mean"),
            评论数量=("sentiment_score", "count")
        )
        .reset_index()
    )

    trend_df["平均情感得分"] = trend_df["平均情感得分"].round(4)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=trend_df["time_group"],
            y=trend_df["平均情感得分"],
            mode="lines+markers",
            line=dict(color=CHART_COLORS["primary"], width=3),
            marker=dict(
                size=8,
                color="#FFFFFF",
                line=dict(color=CHART_COLORS["primary"], width=2)
            ),
            fill="tozeroy",
            fillcolor=CHART_COLORS["primary_light"],
            customdata=trend_df[["评论数量"]],
            hovertemplate=(
                "时间：%{x}<br>"
                "平均情感得分：%{y}<br>"
                "评论数量：%{customdata[0]} 条<extra></extra>"
            )
        )
    )

    fig = apply_clean_plotly_layout(
        fig,
        title="平均情感得分时间趋势",
        height=430
    )

    fig.update_layout(
        showlegend=False,
        xaxis_title=x_title,
        yaxis_title="平均情感得分"
    )

    fig.update_yaxes(range=[0, 1])

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "displayModeBar": False,
            "responsive": True
        }
    )


def render_sentiment_pie_plotly(df: pd.DataFrame):
    """
    动态图：情感环形图，高级浅色版。
    """
    if "sentiment_type" not in df.columns:
        st.warning("缺少 sentiment_type 字段，无法绘制情感环形图。")
        return

    plot_df = (
        df["sentiment_type"]
        .value_counts()
        .reindex(["正向", "中立", "负向"], fill_value=0)
        .reset_index()
    )

    plot_df.columns = ["情感类型", "评论数量"]

    color_map = {
        "正向": CHART_COLORS["positive"],
        "中立": CHART_COLORS["neutral"],
        "负向": CHART_COLORS["negative"]
    }

    fig = go.Figure(
        data=[
            go.Pie(
                labels=plot_df["情感类型"],
                values=plot_df["评论数量"],
                hole=0.58,
                marker=dict(
                    colors=[
                        color_map.get(x, "#64748B")
                        for x in plot_df["情感类型"]
                    ],
                    line=dict(color="#FFFFFF", width=3)
                ),
                textinfo="label+percent",
                textfont=dict(size=14, color=CHART_COLORS["text"]),
                hovertemplate=(
                    "情感类型：%{label}<br>"
                    "评论数量：%{value} 条<br>"
                    "占比：%{percent}<extra></extra>"
                )
            )
        ]
    )

    fig = apply_clean_plotly_layout(
        fig,
        title="评论情感结构占比",
        height=430
    )

    fig.update_layout(
        showlegend=True,
        legend=dict(
            orientation="v",
            x=0.86,
            y=0.5,
            xanchor="left",
            yanchor="middle"
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "displayModeBar": False,
            "responsive": True
        }
    )

def render_high_influence_comments(df: pd.DataFrame, top_n: int = 8):
    required_columns = ["like_count", "influence_score"]

    for column in required_columns:
        if column not in df.columns:
            st.warning(f"缺少字段：{column}，无法展示高影响力评论。")
            return

    text_column = None

    for possible_column in ["clean_text", "comment_text", "original_comment_text"]:
        if possible_column in df.columns:
            text_column = possible_column
            break

    if text_column is None:
        st.warning("缺少评论文本字段，无法展示高影响力评论。")
        return

    top_df = (
        df.sort_values(by="influence_score", ascending=False)
        .head(top_n)
        .copy()
    )

    if top_df.empty:
        st.info("暂无高影响力评论。")
        return

    for index, (_, row) in enumerate(top_df.iterrows(), start=1):
        comment_text = str(row.get(text_column, "")).strip()

        if not comment_text or comment_text.lower() == "nan":
            comment_text = "该评论文本为空或未成功读取。"

        try:
            like_count = int(float(row.get("like_count", 0)))
        except Exception:
            like_count = 0

        try:
            influence_score = round(float(row.get("influence_score", 0)), 4)
        except Exception:
            influence_score = row.get("influence_score", 0)

        comment_text = html.escape(comment_text)

        st.markdown(
            f"""
            <div class="hi-comment-card">
                <div class="hi-comment-title">高影响力评论 {index}</div>
                <div class="hi-comment-text">{comment_text}</div>
                <div class="hi-comment-tags">
                    <span class="hi-tag-like">点赞数：{like_count}</span>
                    <span class="hi-tag-score">影响力分数：{influence_score}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


@st.dialog("完整舆情分析报告")
def show_report_dialog(report_text: str):
    st.markdown(
        """
        <div style="
            color:#64748B;
            font-size:14px;
            line-height:1.7;
            margin-bottom:12px;
        ">
        可在下方滚动查看完整报告内容，如需保存，可点击底部按钮下载。
        </div>
        """,
        unsafe_allow_html=True
    )

    st.text_area(
        label="报告内容",
        value=report_text,
        height=520,
        label_visibility="collapsed"
    )

    col1, col2 = st.columns(2)

    with col1:
        st.download_button(
            label="保存报告到本地",
            data=safe_download_bytes(report_text),
            file_name="final_analysis_report.txt",
            mime="text/plain;charset=utf-8",
            use_container_width=True
        )

    with col2:
        if st.button("关闭窗口", use_container_width=True):
            st.rerun()


# =========================
# 核心流程
# =========================
def crawl_and_clean(video_input, target_valid_count, crawl_mode, status_area, log_lines):
    crawler = BilibiliCommentCrawler()
    cleaner = CommentCleaner()

    start_time = time.time()

    bvid = crawler.extract_bvid(video_input)
    add_log(log_lines, f"识别到 BV 号：{bvid}")
    update_status(status_area, log_lines, 1, 0, "评论采集与清洗", None)

    aid = crawler.get_video_aid(bvid)
    max_pages = math.ceil(MAX_RAW_COMMENTS / PAGE_SIZE)

    raw_comments = []
    df_clean = pd.DataFrame()

    next_page = 0
    use_old_api = False

    add_log(log_lines, f"目标有效评论数量：{target_valid_count} 条")
    add_log(log_lines, f"采集模式：{crawl_mode}")
    add_log(log_lines, f"原始评论保护上限：{MAX_RAW_COMMENTS} 条")

    for page in range(1, max_pages + 1):
        current_valid_count = len(df_clean)
        percent = min(int(current_valid_count / target_valid_count * 100), 100)
        remain_seconds = estimate_remaining_seconds(
            current_valid_count,
            target_valid_count,
            start_time
        )

        update_status(status_area, log_lines, 1, percent, f"正在爬取第 {page} 页评论", remain_seconds)
        add_log(log_lines, f"正在爬取第 {page} 页评论……")

        if len(raw_comments) >= MAX_RAW_COMMENTS:
            add_log(log_lines, "已达到最大原始评论爬取上限，停止爬取。")
            break

        try:
            if not use_old_api:
                page_comments, next_page, is_end = crawler.fetch_main_page(
                    bvid=bvid,
                    aid=aid,
                    next_page=next_page,
                    crawl_mode=crawl_mode
                )

                if not page_comments and page == 1:
                    add_log(log_lines, "新版接口未获取到评论，切换为旧版接口。")
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
                add_log(log_lines, "当前页未获取到评论，停止爬取。")
                break

            remain_count = MAX_RAW_COMMENTS - len(raw_comments)
            raw_comments.extend(page_comments[:remain_count])

            df_clean = build_clean_dataframe(raw_comments, cleaner)

            current_valid_count = len(df_clean)
            percent = min(int(current_valid_count / target_valid_count * 100), 100)
            remain_seconds = estimate_remaining_seconds(
                current_valid_count,
                target_valid_count,
                start_time
            )

            add_log(
                log_lines,
                f"当前原始评论数量：{len(raw_comments)} 条；当前有效评论数量：{current_valid_count} 条"
            )

            update_status(status_area, log_lines, 1, percent, "评论采集与清洗", remain_seconds)

            if current_valid_count >= target_valid_count:
                add_log(log_lines, "已达到目标有效评论数量，停止爬取。")
                update_status(status_area, log_lines, 1, 100, "评论采集与清洗完成", 0)
                break

            if is_end:
                add_log(log_lines, "已经到达评论最后一页，停止爬取。")
                break

            time.sleep(1)

        except Exception as error:
            add_log(log_lines, f"爬取第 {page} 页时发生错误：{error}")
            break

    if df_clean.empty:
        raise RuntimeError("未获得有效评论，请尝试更换视频或降低目标评论数量。")

    if len(df_clean) > target_valid_count:
        df_clean = df_clean.head(target_valid_count).copy()

    raw_save_path = DATA_DIR / "raw_comments.csv"
    clean_save_path = DATA_DIR / "clean_comments.csv"

    save_raw_comments(crawler, raw_comments, raw_save_path)
    save_clean_dataframe(df_clean, clean_save_path)

    add_log(log_lines, f"原始评论已保存：{raw_save_path}")
    add_log(log_lines, f"清洗后评论已保存：{clean_save_path}")
    add_log(log_lines, f"最终有效评论数量：{len(df_clean)} 条")
    add_log(log_lines, "评论采集与清洗完成。")

    return str(clean_save_path)


def run_full_pipeline(video_input, target_valid_count, crawl_mode):
    ensure_dirs()

    status_area = st.empty()
    log_lines = []

    # 日志区域提前创建，运行过程中也能看到
    with st.expander("查看完整进度日志", expanded=False):
        log_area = st.empty()

    st.session_state["log_area"] = log_area

    add_log(log_lines, "正在执行舆情分析流程，请稍等……")
    update_status(status_area, log_lines, 1, 0, "准备开始分析", None)

    clean_path = crawl_and_clean(
        video_input=video_input,
        target_valid_count=target_valid_count,
        crawl_mode=crawl_mode,
        status_area=status_area,
        log_lines=log_lines
    )

    update_status(status_area, log_lines, 2, 70, "情感分析与 AI 复核", None)
    add_log(log_lines, "【2/4】情感分析与 AI 关键评论复核")

    analyzer = SentimentAnalyzer()
    analyzer.run(
        clean_path=clean_path,
        sentiment_save_path=str(DATA_DIR / "sentiment_comments.csv"),
        summary_save_path=str(OUTPUT_DIR / "sentiment_summary.txt")
    )

    add_log(log_lines, "情感分析与 AI 关键评论复核完成。")
    update_status(status_area, log_lines, 3, 85, "生成舆情可视化图表", None)

    visualizer = CommentVisualizer()
    visualizer.run(
        sentiment_path=str(DATA_DIR / "sentiment_comments.csv"),
        output_dir=str(OUTPUT_DIR)
    )

    add_log(log_lines, "舆情可视化图表生成完成。")
    update_status(status_area, log_lines, 4, 95, "生成舆情分析报告", None)

    generator = ReportGenerator()
    generator.run()

    add_log(log_lines, "最终舆情分析报告生成完成。")
    add_log(log_lines, "舆情分析流程已全部完成。")

    # stage=5：让四个阶段全部显示“已完成”
    update_status(status_area, log_lines, 5, 100, "舆情分析完成", 0)

# =========================
# 页面头部
# =========================
st.markdown(
    """
    <div class="hero">
        <div class="hero-title">B站评论舆情分析系统</div>
        <div class="hero-en">Public Opinion Insight Dashboard</div>
        <div class="hero-subtitle">
            基于评论采集、SnowNLP 情感初判、Gemini AI 关键评论复核与可视化分析的舆情洞察平台。
        </div>
        <span class="tag">评论采集</span>
        <span class="tag">情感分析</span>
        <span class="tag">AI复核</span>
        <span class="tag">趋势分析</span>
        <span class="tag">报告生成</span>
    </div>
    """,
    unsafe_allow_html=True
)


# =========================
# 侧边栏输入
# =========================
with st.sidebar:
    st.markdown('<div class="sidebar-title">分析参数</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sidebar-desc">输入 B站 视频链接或 BV号，系统将自动完成评论采集、情感分析、AI复核、图表生成和报告输出。</div>',
        unsafe_allow_html=True
    )

    video_input = st.text_input(
        "B站视频链接或 BV号",
        placeholder="例如：https://www.bilibili.com/video/BVxxxxxxx"
    )

    target_valid_count = st.slider(
        "目标有效评论数量",
        min_value=50,
        max_value=300,
        value=100,
        step=10
    )

    crawl_mode_label = st.radio(
        "评论采集模式",
        ["最新评论：适合观察情绪随时间变化", "最热评论：适合分析高赞观点和舆情焦点"],
        index=1
    )

    crawl_mode = "hot" if crawl_mode_label.startswith("最热") else "latest"

    start_button = st.button(
        "开始分析",
        type="primary",
        use_container_width=True
    )

    reset_button = st.button(
        "重新查看结果",
        use_container_width=True
    )


# =========================
# 启动分析
# =========================
if start_button:
    if not video_input.strip():
        st.error("请先输入 B站 视频链接或 BV号。")
    else:
        try:
            run_full_pipeline(
                video_input=video_input.strip(),
                target_valid_count=target_valid_count,
                crawl_mode=crawl_mode
            )
            st.session_state["analysis_done"] = True
            st.markdown(
                """
                <div class="success-box">
                    分析完成！评论数据、图表和报告已全部生成，请查看下方结果。
                </div>
                """,
                unsafe_allow_html=True
            )

        except Exception as error:
            st.session_state["analysis_done"] = False
            st.error(f"分析过程中发生错误：{error}")

if reset_button:
    st.session_state["analysis_done"] = True


# =========================
# 结果展示
# =========================
df_result = load_sentiment_result()

if df_result is not None:
    metrics = get_basic_metrics(df_result)

    st.markdown(
        """
        <div class="section-title">本次分析结果概览</div>
        <div class="section-desc">以下指标基于清洗后的有效评论和最终情感分析结果生成。</div>
        """,
        unsafe_allow_html=True
    )

    row1_col1, row1_col2, row1_col3 = st.columns(3)

    with row1_col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">有效评论数</div>
                <div class="metric-value">{metrics["total"]}</div>
                <div class="metric-note">用于本次舆情分析</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with row1_col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">正向评论占比</div>
                <div class="metric-value">{metrics["positive_ratio"]}%</div>
                <div class="metric-note">整体认可与积极表达</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with row1_col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">负向评论占比</div>
                <div class="metric-value">{metrics["negative_ratio"]}%</div>
                <div class="metric-note">质疑、不满或焦虑表达</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.write("")

    row2_col1, row2_col2, row2_col3 = st.columns(3)

    with row2_col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">平均情感得分</div>
                <div class="metric-value">{metrics["avg_score"]}</div>
                <div class="metric-note">越接近 1 表示越正向</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with row2_col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">争议度指数</div>
                <div class="metric-value">{metrics["dispute_index"]}</div>
                <div class="metric-note">越高表示正负观点越接近</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with row2_col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">AI复核数量</div>
                <div class="metric-value">{metrics["ai_review_count"]}</div>
                <div class="metric-note">关键评论 Gemini 复核</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.write("")

    # =========================
    # 主分析图表
    # =========================
    with st.container(border=True):
        st.markdown(
            """
            <div class="section-title">主分析图表</div>
            <div class="section-desc">主分析图用于支撑舆情结论。每张图表均可单独下载，用于实验报告或答辩展示。</div>
            """,
            unsafe_allow_html=True
        )

        tab1, tab2, tab3, tab4 = st.tabs(
            ["情感类别分布", "情感得分分布", "情绪时间趋势", "高影响力评论"]
        )

        with tab1:
            render_sentiment_type_plotly(df_result)
            download_png_button(
                OUTPUT_DIR / "sentiment_type_bar.png",
                "下载情感类别分布图"
            )

        with tab2:
            render_sentiment_score_hist_plotly(df_result)
            download_png_button(
                OUTPUT_DIR / "sentiment_score_hist.png",
                "下载情感得分分布图"
            )

        with tab3:
            render_sentiment_trend_plotly(df_result)
            download_png_button(
                OUTPUT_DIR / "sentiment_trend.png",
                "下载情绪时间趋势图"
            )

        with tab4:
            st.markdown(
                """
                <div style="color:#334155;font-size:14px;line-height:1.7;margin-bottom:16px;font-weight:700;">
                    高影响力评论按“点赞数 × 情绪极端程度”排序，下面展示评论内容、点赞数和影响力分数。
                </div>
                """,
                unsafe_allow_html=True
            )

            render_high_influence_comments(df_result)

            image_with_download(
                OUTPUT_DIR / "influence_comments_bar.png",
                "高影响力评论排行图",
                "下载高影响力评论排行图"
            )

    st.write("")

    # =========================
    # 补充展示图
    # =========================
    with st.container(border=True):
        st.markdown(
            """
            <div class="section-title">补充展示图</div>
            <div class="section-desc">词云和环形图主要用于增强展示效果，不作为唯一判断依据。每张补充图也可单独下载。</div>
            """,
            unsafe_allow_html=True
        )

        tab5, tab6, tab7 = st.tabs(
            ["情感环形图", "正向评论词云", "负向评论词云"]
        )

        with tab5:
            render_sentiment_pie_plotly(df_result)
            download_png_button(
                OUTPUT_DIR / "sentiment_pie.png",
                "下载情感分布环形图"
            )

        with tab6:
            image_with_download(
                OUTPUT_DIR / "positive_wordcloud.png",
                "正向评论词云",
                "下载正向评论词云"
            )

        with tab7:
            image_with_download(
                OUTPUT_DIR / "negative_wordcloud.png",
                "负向评论词云",
                "下载负向评论词云"
            )

    st.write("")

    # =========================
    # 数据预览
    # =========================
    with st.container(border=True):
        st.markdown(
            """
            <div class="section-title">代表性评论数据预览</div>
            <div class="section-desc">展示情感分析后的部分字段，便于检查评论内容、判断来源和 AI 复核状态。</div>
            """,
            unsafe_allow_html=True
        )

        preview_columns = [
            "clean_text",
            "sentiment_type",
            "sentiment_score",
            "like_count",
            "influence_score",
            "judgment_source",
            "ai_review_status"
        ]

        available_columns = [
            col for col in preview_columns
            if col in df_result.columns
        ]

        st.dataframe(
            df_result[available_columns].head(20),
            use_container_width=True,
            height=420
        )

        csv_bytes = df_result.to_csv(
            index=False,
            encoding="utf-8-sig"
        ).encode("utf-8-sig")

        st.download_button(
            label="下载完整评论分析表格",
            data=csv_bytes,
            file_name="sentiment_comments.csv",
            mime="text/csv",
            use_container_width=True
        )

    st.write("")

    # =========================
    # 报告查看区
    # =========================
    report_text = load_report_text()

    with st.container(border=True):
        st.markdown(
            """
            <div class="section-title">最终舆情分析报告</div>
            <div class="section-desc">下方为本次评论区大众情绪的总体概括。点击按钮可在弹窗中查看完整报告，并按需保存。</div>
            """,
            unsafe_allow_html=True
        )

        overview_text = build_overview_paragraph(metrics)

        st.markdown(
            f"""
            <div class="overview-text">
                {overview_text}
            </div>
            """,
            unsafe_allow_html=True
        )

        if report_text:
            if st.button(
                "查看完整舆情分析报告",
                type="primary",
                use_container_width=True
            ):
                show_report_dialog(report_text)
        else:
            st.warning("暂未生成最终报告，请先点击左侧开始分析。")

else:
    with st.container(border=True):
        st.markdown(
            """
            <div class="section-title">等待开始分析</div>
            <div class="section-desc">
                请在左侧输入 B站 视频链接或 BV号，选择评论数量和采集模式，然后点击“开始分析”。
            </div>
            """,
            unsafe_allow_html=True
        )


st.markdown(
    """
    <div class="footer">
        B站评论舆情分析系统 · Python 期末大作业展示版
    </div>
    """,
    unsafe_allow_html=True
)