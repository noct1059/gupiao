#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run a single-stock TradingAgents-Astock review and push a candidate-screening brief to Feishu."""

from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import requests
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")
BEIJING_TZ = ZoneInfo("Asia/Shanghai")


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


def _choose_stocks(stock_code: str | None, stock_list: str) -> list[str]:
    if stock_code:
        codes = _parse_stock_codes(stock_code)
        if codes:
            return codes
    codes = _parse_stock_codes(stock_list)
    if not codes:
        raise ValueError("No stock code provided and STOCK_LIST is empty.")
    today = datetime.now(BEIJING_TZ)
    return [codes[today.weekday() % len(codes)]]


def _build_config(provider: str, quick_model: str, deep_model: str, results_dir: Path) -> dict[str, Any]:
    from tradingagents.default_config import DEFAULT_CONFIG

    config = DEFAULT_CONFIG.copy()
    config["llm_provider"] = provider
    config["quick_think_llm"] = quick_model
    config["deep_think_llm"] = deep_model
    config["results_dir"] = str(results_dir)
    config["data_vendors"] = {
        "core_stock_apis": "a_stock",
        "technical_indicators": "a_stock",
        "fundamental_data": "a_stock",
        "news_data": "a_stock",
        "signal_data": "a_stock",
    }
    config["max_debate_rounds"] = int(os.getenv("DEEP_REVIEW_DEBATE_ROUNDS", "1"))
    config["max_risk_discuss_rounds"] = int(os.getenv("DEEP_REVIEW_RISK_ROUNDS", "1"))
    config["output_language"] = "Chinese"
    return config


def _select_models(provider: str) -> tuple[str, str]:
    quick = os.getenv("DEEP_REVIEW_QUICK_MODEL", "").strip()
    deep = os.getenv("DEEP_REVIEW_DEEP_MODEL", "").strip()
    if quick and deep:
        return quick, deep

    defaults = {
        "deepseek": ("deepseek-chat", "deepseek-chat"),
        "google": ("gemini-2.5-flash", "gemini-2.5-pro"),
        "qwen": ("qwen-plus", "qwen-max"),
        "minimax": ("MiniMax-M2.7-highspeed", "MiniMax-M2.7"),
        "openai": ("gpt-5.4-mini", "gpt-5.4"),
    }
    default_quick, default_deep = defaults.get(provider, defaults["deepseek"])
    return quick or default_quick, deep or default_deep


def _trim_report(text: str, max_chars: int = 15000) -> str:
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n[内容过长，已截断；完整报告见 GitHub Actions artifact。]"


def _deepseek_chat(api_key: str, model: str, prompt: str) -> str:
    base_url = os.getenv("LLM_DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
    resp = requests.post(
        f"{base_url}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "你是A股主力行为、量化博弈与情绪周期复盘助手。"
                        "你只能根据用户给出的研报内容做条件推演，不得编造研报中没有的数据，"
                        "不得把推演包装成确定事实。"
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": float(os.getenv("DEEP_REVIEW_SCREENING_TEMPERATURE", "0.2")),
        },
        timeout=90,
    )
    resp.raise_for_status()
    body = resp.json()
    return (body.get("choices") or [{}])[0].get("message", {}).get("content", "").strip()


def _build_screening_brief(stock_code: str, trade_date: str, report: str, fallback_model: str) -> str:
    api_key = (
        os.getenv("DEEPSEEK_API_KEY", "").strip()
        or os.getenv("LLM_DEEPSEEK_API_KEY", "").strip()
        or os.getenv("DEEPSEEK_API_KEYS", "").split(",", 1)[0].strip()
    )
    if not api_key:
        return ""

    model = (
        os.getenv("DEEP_REVIEW_SCREENING_MODEL", "").strip()
        or os.getenv("DEEP_REVIEW_DEEP_MODEL", "").strip()
        or fallback_model
    )
    prompt = f"""请根据下面这份A股深度投研报告，用【V11.5 A股主力行为、量化博弈与情绪周期决策系统】做二次复判。

你的任务不是普通技术分析，也不是机构研报打分。
你要站在主力、量化、游资、散户、机构的多方博弈角度，推演：
- 谁正在被迫买入
- 谁正在被迫卖出
- 谁已经失去流动性优势
- 谁正在利用情绪收割

第一原则：
不要先判断涨跌，必须先判断【当前市场阶段】。阶段比位置更重要。

阶段识别只能从以下选择，允许给“主阶段 + 次阶段”：
1. 吸筹期：缩量、阴跌、横盘、跌不动、板块弱但个股抗跌。
2. 试盘期：突然放量、快速脱离平台、回踩不深、分时承接明显。
3. 洗盘期：急跌、恐慌、跌破均线但迅速收回；缩量下跌更像假洗盘，放量持续下跌更像真出货。
4. 主升期：回调越来越浅、放量上涨、板块联动、龙头共振。
5. 派发期：放量滞涨、高频冲高回落、利好不涨、板块强但个股弱。
6. 退潮期：板块走弱、跌破平台、无承接、缩量阴跌、反弹无持续性。

必须重点区分：
- 真破位：放量跌破、次日继续弱、无法快速收回、板块同步走弱、关键均线拐头、资金持续流出。
- 假破位：急跌、缩量、长下影、很快拉回、板块未同步崩塌、恐慌后承接增强。

情绪周期必须判断：
情绪启动 / 情绪高潮 / 情绪分歧 / 情绪退潮 / 情绪冰点 / 修复阶段。

A股语境约束：
1. 趋势分歧不等于趋势死亡。要区分良性分歧、缩量洗盘、放量出货、板块退潮。
2. 基本面瑕疵不能一票否决，除非报告显示硬伤、财务风险、流动性风险或逻辑坍塌。
3. 不要照搬海外机构长期价值口径。A股要更重视板块强度、资金承接、题材持续性、情绪周期、量化假动作。
4. 不要迷信单日资金流，也不要单纯资金流崇拜；资金行为必须结合位置、阶段、量价、板块、情绪。
5. 如果报告缺少分时、成交量或板块数据，必须标注“证据不足”，不能编造。

真正机会：
不是人人看多，而是人人看空但跌不动。

真正风险：
不是人人看空，而是人人看多但涨不动。

仓位系统：
禁止满仓赌博、一把梭哈。必须使用“试错 -> 确认 -> 加减仓”。
- 观察仓：10%-20%，用于试错和观察节奏。
- 确认仓：30%-50%，用于修复确认、放量突破。
- 主升仓：70%以上，仅限趋势明确、板块共振。
- 风险仓：20%以下，用于退潮确认、流动性恶化。

输出要求：
1. 不要只说“看多/看空”，要做条件推演。
2. 不要把主力意图说成确定事实，用“更像/可能/若...则...”表达。
3. 输出要适合飞书快速阅读，先给结论，再给推演。
4. 最终备选结论必须四选一：加入备选 / 小仓观察 / 只观察不碰 / 暂不加入。
5. 输出中文 Markdown。

固定格式：
# 个股备选池复判
- 股票代码：{stock_code}
- 分析日期：{trade_date}
- 备选结论：加入备选 / 小仓观察 / 只观察不碰 / 暂不加入
- 当前阶段：吸筹期 / 试盘期 / 洗盘期 / 主升期 / 派发期 / 退潮期
- 情绪周期：情绪启动 / 情绪高潮 / 情绪分歧 / 情绪退潮 / 情绪冰点 / 修复阶段
- 置信度：高 / 中 / 低

## 1. 当前阶段判断
说明主阶段、次阶段，以及为什么不是其他阶段。

## 2. 主力真实意图推演
从吸筹、洗盘、出货、自救、引导情绪、诱多、诱空中选择更可能的剧本，并说明依据。

## 3. 量化可能收割方向
说明更可能在收割追突破、固定止损、死拿不动，还是暂时看不出明显收割对象。

## 4. 当前是否是假动作
判断是真破位、假破位、假突破、正常分歧，还是证据不足。

## 5. 真正危险点
列出2-4条会让节奏恶化的条件。

## 6. 真正转强条件
列出2-4条能证明资金重新掌控节奏的条件。

## 7. 最优仓位策略
给观察仓、确认仓、风险仓建议；不要建议满仓。

## 8. 短线、中线路径推演
分别用“若...则...”描述，不要做单一路径预测。

## 9. 最可能剧本
用一段话总结当前最可能发生的剧本。

## 10. 最容易被骗的位置
指出散户最容易误判的位置，例如假突破、假破位、缩量阴跌、利好不涨等。

深度投研报告：
{_trim_report(report, max_chars=12000)}
"""
    try:
        return _deepseek_chat(api_key, model, prompt)
    except Exception as exc:
        print(f"Screening brief failed: {exc}", file=sys.stderr)
        return ""


def _feishu_sign(secret: str, timestamp: str) -> str:
    key = f"{timestamp}\n{secret}".encode("utf-8")
    digest = hmac.new(key, b"", digestmod=hashlib.sha256).digest()
    return base64.b64encode(digest).decode("utf-8")


def _send_feishu(content: str) -> bool:
    webhook = os.getenv("FEISHU_WEBHOOK_URL", "").strip()
    if not webhook:
        print("FEISHU_WEBHOOK_URL is not set; skip notification.")
        return True

    keyword = os.getenv("FEISHU_WEBHOOK_KEYWORD", "").strip()
    if keyword and keyword not in content:
        content = f"{keyword}\n\n{content}"

    payload: dict[str, Any] = {"msg_type": "text", "content": {"text": _trim_report(content)}}
    secret = os.getenv("FEISHU_WEBHOOK_SECRET", "").strip()
    if secret:
        timestamp = str(int(time.time()))
        payload["timestamp"] = timestamp
        payload["sign"] = _feishu_sign(secret, timestamp)

    resp = requests.post(webhook, json=payload, timeout=30)
    if resp.status_code >= 400:
        print(f"Feishu send failed: HTTP {resp.status_code} {resp.text}", file=sys.stderr)
        return False
    try:
        body = resp.json()
    except Exception:
        body = {}
    if body and body.get("code", 0) not in (0, None):
        print(f"Feishu send failed: {body}", file=sys.stderr)
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Run TradingAgents-Astock nightly deep review")
    parser.add_argument("--stock-code", default=os.getenv("DEEP_REVIEW_STOCK_CODE", ""))
    parser.add_argument("--trade-date", default=os.getenv("DEEP_REVIEW_TRADE_DATE", ""))
    parser.add_argument("--no-notify", action="store_true")
    args = parser.parse_args()

    stock_list = os.getenv("STOCK_LIST", "")
    stock_codes = _choose_stocks(args.stock_code, stock_list)
    trade_date = args.trade_date or datetime.now(BEIJING_TZ).strftime("%Y-%m-%d")
    provider = os.getenv("DEEP_REVIEW_LLM_PROVIDER", "deepseek").strip().lower()
    quick_model, deep_model = _select_models(provider)

    from tradingagents.graph.trading_graph import TradingAgentsGraph

    reports_dir = ROOT / "reports" / "deep_stock_review"
    reports_dir.mkdir(parents=True, exist_ok=True)
    results_dir = reports_dir / "tradingagents_logs"

    config = _build_config(provider, quick_model, deep_model, results_dir)
    all_contents: list[str] = []
    for stock_code in stock_codes:
        started = time.time()
        graph = TradingAgentsGraph(debug=False, config=config)
        final_state, signal = graph.propagate(stock_code, trade_date)
        elapsed = time.time() - started

        decision = final_state.get("final_trade_decision", "") or str(signal)
        raw_report = decision.strip()
        screening_brief = _build_screening_brief(stock_code, trade_date, raw_report, deep_model)
        header = (
            f"# A股个股备选筛选（{trade_date}）\n\n"
            f"- 股票代码：{stock_code}\n"
            f"- 模型：{provider} / quick={quick_model} / deep={deep_model}\n"
            f"- 耗时：{elapsed / 60:.1f} 分钟\n"
            f"- 原始信号：{signal}\n"
        )
        if screening_brief:
            content = f"{header}\n{screening_brief.strip()}\n\n---\n\n## 原始深度报告\n\n{raw_report}"
        else:
            content = f"{header}\n## 原始深度报告\n\n{raw_report}"

        out_path = reports_dir / f"deep_review_{stock_code}_{trade_date}.md"
        out_path.write_text(content, encoding="utf-8")
        print(content)
        print(f"\nSaved report: {out_path}")
        all_contents.append(content)

    if not args.no_notify and not _send_feishu("\n\n---\n\n".join(all_contents)):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
