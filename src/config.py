"""
EEZO収益シミュレーション - 設定・パラメータ読み込みモジュール
"""
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import matplotlib.pyplot as plt


# 日本語フォント設定
plt.rcParams["font.family"] = "IPAGothic"
plt.rcParams["axes.unicode_minus"] = False

# プロジェクトルート
PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "exp001_baseline" / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"


def load_assumptions() -> Dict[str, Any]:
    """
    assumptions.csvからパラメータを読み込む

    Returns:
        パラメータをカテゴリ別に整理した辞書
    """
    df = pd.read_csv(DATA_RAW / "assumptions.csv")

    params: Dict[str, Any] = {}

    for _, row in df.iterrows():
        category = row["category"]
        param = row["parameter"]
        value = row["value"]

        if category not in params:
            params[category] = {}
        params[category][param] = value

    return params


# 定数（CLAUDE.mdより）
TARGET_SALES = 10_000_000  # 年間目標売上
GROSS_MARGIN_RATE = 0.15  # 粗利率
TARGET_GROSS_PROFIT = 1_500_000  # 目標粗利

# 価格帯設定
TIER_A_PRICE = 5_000  # 5,000円帯
TIER_A_PROFIT = 750  # 粗利/個
TIER_B_PRICE = 10_000  # 10,000円帯
TIER_B_PROFIT = 1_500  # 粗利/個

# チャネル情報
CHANNELS = {
    "旅客メルマガ": {
        "current_cvr": 0.0004,
        "target_cvr": 0.005,
        "annual_traffic": 400_000,
    },
    "O2O（船内→EC）": {
        "current_cvr": 0.02,
        "target_cvr": 0.03,
        "annual_traffic": 50_000,
    },
    "SNS直接流入": {
        "current_cvr": 0.005,
        "target_cvr": 0.01,
        "annual_traffic": 100_000,
    },
    "同梱→リピート": {
        "current_cvr": 0.10,
        "target_cvr": 0.20,
        "annual_traffic": 2_000,
    },
}
