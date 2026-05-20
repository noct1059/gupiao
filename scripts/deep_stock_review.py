#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run a nightly single-stock TradingAgents-Astock deep review and push it to Feishu."""

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
        content = (
            f"# A股盘后深度复盘（{trade_date}）\n\n"
            f"- 股票代码：{stock_code}\n"
            f"- 模型：{provider} / quick={quick_model} / deep={deep_model}\n"
            f"- 耗时：{elapsed / 60:.1f} 分钟\n"
            f"- 信号：{signal}\n\n"
            f"{decision.strip()}"
        )

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
