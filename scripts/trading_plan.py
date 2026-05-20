#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate A-share trading plan pushes for pre/intraday/post-market windows."""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")
BEIJING_TZ = ZoneInfo("Asia/Shanghai")


@dataclass(frozen=True)
class PlanMode:
    key: str
    title: str
    objective: str


PLAN_MODES = {
    "pre-market": PlanMode("pre-market", "A股盘前计划", "建立今日看盘框架、主线候选和自选股观察条件"),
    "midday": PlanMode("midday", "A股早盘观察", "验证开盘后资金方向、热点强度和盘前预案"),
    "afternoon": PlanMode("afternoon", "A股午后确认", "判断主线延续、轮动退潮和尾盘风险"),
    "post-market": PlanMode("post-market", "A股盘后复盘", "复盘市场性质、资金主线和明日计划"),
}


def _safe_head(records: Any, limit: int = 8) -> list[dict[str, Any]]:
    try:
        return records.head(limit).fillna("").to_dict("records")
    except Exception:
        return []


def _sort_by(df: Any, column: str, ascending: bool = False) -> Any:
    try:
        if column in df.columns:
            return df.sort_values(by=column, ascending=ascending)
    except Exception:
        pass
    return df


def _sort_by_first(df: Any, columns: Iterable[str], ascending: bool = False) -> Any:
    column = _first_existing_column(df, columns)
    return _sort_by(df, column, ascending=ascending) if column else df


def _pick(row: dict[str, Any], names: Iterable[str], default: Any = "") -> Any:
    for name in names:
        if name in row and row[name] not in (None, ""):
            return row[name]
    return default


def _format_records(title: str, rows: list[dict[str, Any]], columns: list[tuple[str, list[str]]]) -> str:
    if not rows:
        return f"### {title}\n暂无可用数据\n"

    lines = [f"### {title}"]
    for idx, row in enumerate(rows, 1):
        parts = []
        for label, keys in columns:
            value = _pick(row, keys)
            if value != "":
                parts.append(f"{label}:{value}")
        if parts:
            lines.append(f"{idx}. " + " | ".join(parts))
    return "\n".join(lines) + "\n"


def _format_optional_number(value: Any, suffix: str = "") -> str:
    number = _as_float(value)
    if number is None:
        return str(value or "缺失")
    return f"{number:.2f}{suffix}"


def _as_float(value: Any) -> float | None:
    try:
        if value in (None, "", "-", "--"):
            return None
        return float(str(value).replace("%", "").replace(",", ""))
    except Exception:
        return None


def _amount_yi(value: Any) -> str:
    number = _as_float(value)
    if number is None:
        return str(value or "")
    return f"{number / 100000000:.1f}亿"


def _first_existing_column(df: Any, names: Iterable[str]) -> str | None:
    try:
        for name in names:
            if name in df.columns:
                return name
    except Exception:
        pass
    return None


def _summarize_breadth(spot: Any) -> str:
    try:
        pct_col = "涨跌幅"
        amount_col = _first_existing_column(spot, ["成交额", "成交金额"])
        if pct_col not in spot.columns:
            return "### 全市场温度\n缺少涨跌幅字段，无法统计涨跌家数。\n"

        pct = spot[pct_col].apply(_as_float)
        up_count = int((pct > 0).sum())
        down_count = int((pct < 0).sum())
        flat_count = int((pct == 0).sum())
        strong_count = int((pct >= 5).sum())
        weak_count = int((pct <= -5).sum())
        total_count = int(pct.notna().sum())

        lines = [
            "### 全市场温度",
            f"- 股票覆盖数：{total_count}",
            f"- 上涨/下跌/平盘：{up_count}/{down_count}/{flat_count}",
            f"- 涨幅>=5% / 跌幅<=-5%：{strong_count}/{weak_count}",
        ]
        if amount_col:
            amount = spot[amount_col].apply(_as_float).dropna().sum()
            lines.append(f"- 全市场成交额估算：{_amount_yi(amount)}")
        return "\n".join(lines) + "\n"
    except Exception as exc:
        return f"### 全市场温度\n获取失败：{exc}\n"


def _get_today_zt_pool(ak: Any) -> Any:
    today = datetime.now(BEIJING_TZ).strftime("%Y%m%d")
    return ak.stock_zt_pool_em(date=today)


def _parse_stock_codes(stock_list: str) -> list[str]:
    seen: set[str] = set()
    codes: list[str] = []
    for token in re.split(r"[,，\s;；]+", stock_list or ""):
        match = re.search(r"\d{6}", token)
        if not match:
            continue
        code = match.group(0)
        if code not in seen:
            seen.add(code)
            codes.append(code)
    return codes


def _collect_watchlist_snapshot(stock_list: str) -> str:
    codes = _parse_stock_codes(stock_list)
    if not codes:
        return "### 自选股实时快照\n未配置自选股代码。\n"

    try:
        from data_provider.akshare_fetcher import AkshareFetcher
    except Exception as exc:
        return f"### 自选股实时快照\n获取失败：AkshareFetcher 不可用：{exc}\n"

    fetcher = AkshareFetcher()
    lines = [
        "### 自选股实时快照",
        f"- 快照时间：{datetime.now(BEIJING_TZ).strftime('%Y-%m-%d %H:%M:%S')} 北京时间",
        "- 说明：个股涨跌以本表为准；若本表缺失，不要编造个股涨跌幅。",
    ]
    for code in codes:
        quote = None
        errors: list[str] = []
        for source in ("tencent", "sina", "em"):
            try:
                quote = fetcher.get_realtime_quote(code, source=source)
                if quote:
                    break
                errors.append(f"{source}: 空")
            except Exception as exc:
                errors.append(f"{source}: {type(exc).__name__}")

        if quote:
            source_name = getattr(getattr(quote, "source", None), "value", None) or str(getattr(quote, "source", ""))
            lines.append(
                "- "
                + " | ".join(
                    [
                        f"代码:{code}",
                        f"名称:{quote.name or '缺失'}",
                        f"最新价:{_format_optional_number(quote.price)}",
                        f"涨跌幅:{_format_optional_number(quote.change_pct, '%')}",
                        f"涨跌额:{_format_optional_number(quote.change_amount)}",
                        f"成交额:{_amount_yi(quote.amount) if quote.amount is not None else '缺失'}",
                        f"来源:{source_name}",
                    ]
                )
            )
        else:
            lines.append(f"- 代码:{code} | 获取失败：{'；'.join(errors[-3:])}")
    return "\n".join(lines) + "\n"


def _collect_index_snapshot(ak: Any) -> str:
    try:
        source_note = "东方财富"
        try:
            index_df = ak.stock_zh_index_spot_em()
        except Exception as primary_exc:
            index_df = ak.stock_zh_index_spot_sina()
            source_note = f"新浪备用（东方财富失败：{primary_exc}）"
        wanted = {"上证指数", "深证成指", "创业板指", "科创50", "沪深300", "中证500", "中证1000"}
        rows = []
        for row in index_df.fillna("").to_dict("records"):
            name = str(_pick(row, ["名称", "name"]))
            code = str(_pick(row, ["代码", "code"]))
            if name in wanted or code in {"000001", "399001", "399006", "000688", "000300", "000905", "000852"}:
                rows.append(row)
        if not rows:
            rows = _safe_head(index_df, 8)
        return f"### 数据源说明\n核心指数快照来源：{source_note}\n\n" + _format_records(
            "核心指数快照",
            rows,
            [
                ("代码", ["代码", "code"]),
                ("名称", ["名称", "name"]),
                ("最新", ["最新价", "最新"]),
                ("涨跌幅", ["涨跌幅"]),
                ("成交额", ["成交额", "成交金额"]),
                ("振幅", ["振幅"]),
            ],
        )
    except Exception as exc:
        return f"### 核心指数快照\n获取失败：{exc}\n"


def _collect_external_context(mode: PlanMode, search_context_hint: str = "") -> str:
    label = "盘前重点" if mode.key == "pre-market" else "外部环境"
    return (
        f"### {label}\n"
        "- 外围指数、人民币汇率、商品期货暂由新闻搜索结果辅助判断。\n"
        "- 如果搜索结果不足，请在结论中明确写“外部环境信号不足”，不要硬推断。\n"
        f"- 当前模式：{mode.title}。{search_context_hint}\n"
    )


def collect_akshare_snapshot(mode: PlanMode, stock_list: str = "") -> str:
    """Collect optional A-share breadth, hot stock, sector and fund-flow data."""
    blocks: list[str] = []
    try:
        import akshare as ak
    except Exception as exc:
        return f"AkShare 不可用：{exc}"

    blocks.append(_collect_external_context(mode))
    blocks.append(_collect_watchlist_snapshot(stock_list))
    blocks.append(_collect_index_snapshot(ak))

    spot = None
    try:
        spot_source = "东方财富"
        try:
            spot = ak.stock_zh_a_spot_em()
        except Exception as primary_exc:
            spot = ak.stock_zh_a_spot()
            spot_source = f"新浪备用（东方财富失败：{primary_exc}）"
        blocks.append(f"### 数据源说明\n全市场股票快照来源：{spot_source}\n")
        blocks.append(_summarize_breadth(spot))
        rows = _safe_head(_sort_by(spot, "涨跌幅"), 10)
        blocks.append(
            _format_records(
                "热门股票/涨幅前列",
                rows,
                [
                    ("代码", ["代码", "code"]),
                    ("名称", ["名称", "name"]),
                    ("涨跌幅", ["涨跌幅"]),
                    ("成交额", ["成交额"]),
                    ("换手", ["换手率"]),
                ],
            )
        )

        rows = _safe_head(_sort_by_first(spot, ["成交额", "成交金额"]), 10)
        blocks.append(
            _format_records(
                "成交额前列股票",
                rows,
                [
                    ("代码", ["代码", "code"]),
                    ("名称", ["名称", "name"]),
                    ("涨跌幅", ["涨跌幅"]),
                    ("成交额", ["成交额", "成交金额"]),
                    ("换手", ["换手率"]),
                ],
            )
        )
    except Exception as exc:
        blocks.append(f"### 全市场股票快照\n获取失败：{exc}\n")

    try:
        industry = ak.stock_board_industry_name_em()
        rows = _safe_head(_sort_by(industry, "涨跌幅"), 8)
        blocks.append(
            _format_records(
                "热门行业板块",
                rows,
                [
                    ("板块", ["板块名称", "名称"]),
                    ("涨跌幅", ["涨跌幅"]),
                    ("上涨家数", ["上涨家数"]),
                    ("下跌家数", ["下跌家数"]),
                    ("领涨股", ["领涨股票"]),
                ],
            )
        )
    except Exception as exc:
        try:
            zt_pool_for_industry = _get_today_zt_pool(ak)
            industry_col = _first_existing_column(zt_pool_for_industry, ["所属行业"])
            if industry_col:
                grouped = (
                    zt_pool_for_industry[industry_col]
                    .fillna("")
                    .astype(str)
                    .replace("", "未分类")
                    .value_counts()
                    .head(8)
                    .reset_index()
                )
                grouped.columns = ["行业", "涨停家数"]
                blocks.append(
                    "### 热门行业板块（降级：涨停池行业分布）\n"
                    f"主板块接口获取失败：{exc}\n"
                    + _format_records(
                        "涨停行业分布",
                        grouped.to_dict("records"),
                        [
                            ("行业", ["行业"]),
                            ("涨停家数", ["涨停家数"]),
                        ],
                    )
                )
            else:
                blocks.append(f"### 热门行业板块\n获取失败：{exc}\n")
        except Exception as fallback_exc:
            blocks.append(f"### 热门行业板块\n获取失败：{exc}；涨停池行业降级也失败：{fallback_exc}\n")

    try:
        concept = ak.stock_board_concept_name_em()
        rows = _safe_head(_sort_by(concept, "涨跌幅"), 8)
        blocks.append(
            _format_records(
                "热门概念板块",
                rows,
                [
                    ("概念", ["板块名称", "名称"]),
                    ("涨跌幅", ["涨跌幅"]),
                    ("上涨家数", ["上涨家数"]),
                    ("下跌家数", ["下跌家数"]),
                    ("领涨股", ["领涨股票"]),
                ],
            )
        )
    except Exception as exc:
        blocks.append(f"### 热门概念板块\n获取失败：{exc}\n")

    try:
        fund_flow = ak.stock_fund_flow_industry(symbol="即时")
        rows = _safe_head(fund_flow, 8)
        blocks.append(
            _format_records(
                "行业资金流向",
                rows,
                [
                    ("行业", ["行业", "名称"]),
                    ("净流入", ["净流入", "主力净流入-净额"]),
                    ("净占比", ["净占比", "主力净流入-净占比"]),
                    ("涨跌幅", ["涨跌幅"]),
                ],
            )
        )
    except Exception as exc:
        blocks.append(f"### 行业资金流向\n获取失败：{exc}\n")

    try:
        zt_pool = _get_today_zt_pool(ak)
        rows = _safe_head(zt_pool, 12)
        blocks.append(
            _format_records(
                "涨停池",
                rows,
                [
                    ("代码", ["代码"]),
                    ("名称", ["名称"]),
                    ("涨跌幅", ["涨跌幅"]),
                    ("封板资金", ["封板资金"]),
                    ("首次封板", ["首次封板时间"]),
                    ("连板数", ["连板数"]),
                ],
            )
        )
    except Exception as exc:
        blocks.append(
            "### 涨停池\n"
            f"获取失败：{exc}\n"
            "提示：非交易日或接口临时不可用时，涨停池可能为空。\n"
        )

    return "\n".join(blocks)


def build_search_context(mode: PlanMode, search_service: Any) -> str:
    if not search_service or not search_service.is_available:
        return "搜索服务不可用。"

    queries = [
        "A股 今日 盘面 热点 板块 资金流向",
        "A股 今日 涨停 跌停 连板 热点 题材",
        "A股 今日 政策 消息 产业 催化",
        "今日 上证指数 深证成指 创业板指 成交额 涨跌家数",
    ]
    if mode.key == "pre-market":
        queries.extend(
            [
                "A股 今日 盘前 外围市场 人民币 商品 政策",
                "隔夜 美股 港股 A50 人民币 原油 黄金 对A股影响",
            ]
        )
    elif mode.key in {"midday", "afternoon"}:
        queries.append("A股 盘中 主力资金 热门股票 板块异动")
    else:
        queries.append("A股 收盘 复盘 主线 板块 资金")

    blocks: list[str] = []
    for query in queries:
        try:
            response = search_service.search(query, max_results=5, days=2)
            lines = [f"### 搜索：{query}"]
            if response.success and response.results:
                for idx, item in enumerate(response.results[:5], 1):
                    title = getattr(item, "title", "")
                    snippet = getattr(item, "snippet", "")
                    source = getattr(item, "source", "")
                    published = getattr(item, "published_date", "")
                    lines.append(f"{idx}. {title} | {source} | {published}\n   {snippet}")
            else:
                lines.append(f"无结果：{getattr(response, 'error_message', '')}")
            blocks.append("\n".join(lines))
        except Exception as exc:
            blocks.append(f"### 搜索：{query}\n获取失败：{exc}")
    return "\n\n".join(blocks)


def build_prompt(mode: PlanMode, stock_list: str, market_data: str, search_context: str) -> str:
    now = datetime.now(BEIJING_TZ)
    weekday_names = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
    weekday = weekday_names[now.weekday()]
    trading_day_note = "非交易日，盘面数据可能来自最近一个交易日" if now.weekday() >= 5 else "交易日"
    common_rules = """
你是一个偏交易计划型的 A 股复盘助手。请严格遵守：
1. 不给绝对买卖建议，不说必涨必跌，不鼓励追高。
2. 所有判断必须绑定数据、新闻、资金或盘面现象。
3. 输出要短、清楚、可验证，避免泛泛而谈。
4. 对个股只给：观察角色、确认信号、失效信号、风险等级。
5. 如果数据缺失，明确写“数据缺失，不作为判断依据”。
6. 优先使用“核心指数快照、全市场温度、行业资金流向、热门板块、涨停池、成交额前列股票”这些硬数据。
7. 非交易日或数据源失败时，要区分“市场本身无交易”和“接口获取失败”，不要把缺失数据当成利空。
8. 严禁编造日期、指数点位、成交额、涨停数量、新闻事件；没有数据就写缺失。
"""

    mode_instructions = {
        "pre-market": """
输出结构：
【A股盘前计划】
一、今日市场环境：3-5 条，只写会影响今日风险偏好的因素。
二、今日大盘观察框架：指数、量能、情绪、风险点。
三、今日主线候选：2-4 条，每条写触发因素、核心板块/股票、验证条件、失效条件。
四、自选股观察计划：逐只写今日角色、确认信号、失效信号。
五、今日纪律：3 条以内。
""",
        "midday": """
输出结构：
【A股早盘观察】
一、盘面温度：指数、成交额、涨跌家数、涨停跌停。
二、资金动向：流入板块、流出板块、权重/题材风格。
三、热点验证：盘前主线是否被验证，有没有新主线。
四、自选股状态：逐只写强弱、确认/未确认/失效。
五、午后观察：3 条以内。
""",
        "afternoon": """
输出结构：
【A股午后确认】
一、午后盘面性质：进攻、轮动、防守或退潮。
二、资金与热点：延续的方向、回落的方向、新异动方向。
三、热门股票作用：只说明它们代表的板块强弱，不追涨。
四、自选股处理框架：逐只写继续观察/降低关注/等待确认。
五、尾盘风险：3 条以内。
""",
        "post-market": """
输出结构：
【A股盘后复盘】
一、今日市场定性：强修复、弱修复、震荡、防守或退潮，并给依据。
二、今日真正主线：1-3 条，写强的原因、核心股票、持续性。
三、资金与情绪：资金流入/流出、涨停跌停、连板/高位反馈。
四、自选股复盘：逐只写确认/未确认/失效和明日观察条件。
五、明日计划：优先方向、不碰方向、大盘确认信号、风险提示。
""",
    }

    return f"""{common_rules}

当前日期：{now.strftime("%Y-%m-%d")}（{weekday}，{trading_day_note}）
当前任务：{mode.title}
任务目标：{mode.objective}
自选股：{stock_list}

{mode_instructions[mode.key]}

以下是盘面数据：
{market_data}

以下是新闻/资讯搜索结果：
{search_context}

请直接输出飞书可读的 Markdown 正文，控制在 1200-2200 字。
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate A-share trading plan")
    parser.add_argument("--mode", choices=sorted(PLAN_MODES), default=os.getenv("TRADING_PLAN_MODE", "pre-market"))
    parser.add_argument("--no-notify", action="store_true", help="Print only, do not send notification")
    args = parser.parse_args()

    mode = PLAN_MODES[args.mode]

    from src.analyzer import GeminiAnalyzer
    from src.config import get_config
    from src.core.market_review_runtime import build_market_review_runtime

    config = get_config()
    stock_list = getattr(config, "stock_list", None) or os.getenv("STOCK_LIST", "")
    notifier, analyzer, search_service = build_market_review_runtime(config)
    if analyzer is None:
        analyzer = GeminiAnalyzer(config=config)

    market_data = collect_akshare_snapshot(mode, stock_list)
    search_context = build_search_context(mode, search_service)
    prompt = build_prompt(mode, stock_list, market_data, search_context)

    text = analyzer.generate_text(prompt, max_tokens=4096, temperature=0.3) if analyzer else None
    if not text:
        text = f"【{mode.title}】\n\n生成失败：LLM 未返回内容。请检查 DeepSeek 配置和运行日志。"

    timestamp = datetime.now(BEIJING_TZ).strftime("%Y-%m-%d %H:%M")
    content = f"# {mode.title}（{timestamp}）\n\n{text.strip()}"
    print(content)

    if not args.no_notify:
        ok = notifier.send(content, email_send_to_all=True, route_type="report")
        if not ok:
            print("Notification send failed", file=sys.stderr)
            return 2

    reports_dir = ROOT / "reports"
    reports_dir.mkdir(exist_ok=True)
    out = reports_dir / f"trading_plan_{mode.key}_{datetime.now(BEIJING_TZ).strftime('%Y%m%d_%H%M%S')}.md"
    out.write_text(content, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
