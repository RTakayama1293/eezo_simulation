"""
シミュレーション項目4: 投資対効果（同梱カード）
同梱カード投資額 vs リピート率改善による粗利増分を分析
"""
from typing import Tuple, List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import (
    TIER_A_PRICE,
    TIER_A_PROFIT,
    TIER_B_PRICE,
    TIER_B_PROFIT,
    FIGURES_DIR,
)


# 同梱カードパラメータ
CARD_COST_LOW = 50  # 円/枚
CARD_COST_HIGH = 80  # 円/枚
CARD_VOLUME = 2000  # 枚（基準）

# リピート関連
REPEAT_BASE_CUSTOMERS = 2000  # 既存顧客ベース
REPEAT_CURRENT_CVR = 0.10  # 現状リピート率10%
REPEAT_TARGET_CVR = 0.20  # 目標リピート率20%

# 平均値
AVG_PRICE = (TIER_A_PRICE + TIER_B_PRICE) / 2
AVG_PROFIT = (TIER_A_PROFIT + TIER_B_PROFIT) / 2


def calculate_roi_curve() -> pd.DataFrame:
    """
    同梱カード投資のROI曲線を計算

    Returns:
        ROI分析結果DataFrame
    """
    results: List[Dict] = []

    # リピート率の改善幅を変化させる（0%〜15%改善）
    repeat_improvements = np.arange(0, 0.16, 0.01)

    for card_cost in [CARD_COST_LOW, CARD_COST_HIGH]:
        investment = card_cost * CARD_VOLUME

        for improvement in repeat_improvements:
            new_repeat_rate = REPEAT_CURRENT_CVR + improvement

            # リピート購入数の増加
            base_repeat_sales = REPEAT_BASE_CUSTOMERS * REPEAT_CURRENT_CVR
            new_repeat_sales = REPEAT_BASE_CUSTOMERS * new_repeat_rate
            incremental_sales = new_repeat_sales - base_repeat_sales

            # 粗利増分
            incremental_profit = incremental_sales * AVG_PROFIT

            # ROI
            roi = (incremental_profit - investment) / investment * 100 if investment > 0 else 0

            # 損益
            net_profit = incremental_profit - investment

            results.append({
                "カード単価": f"{card_cost}円",
                "投資額": investment,
                "リピート率改善": f"+{improvement*100:.0f}%",
                "リピート率": f"{new_repeat_rate*100:.0f}%",
                "追加販売数": int(incremental_sales),
                "粗利増分": int(incremental_profit),
                "純利益": int(net_profit),
                "ROI": roi,
            })

    return pd.DataFrame(results)


def find_breakeven() -> Dict:
    """
    損益分岐点を算出

    Returns:
        損益分岐点情報の辞書
    """
    breakeven = {}

    for card_cost in [CARD_COST_LOW, CARD_COST_HIGH]:
        investment = card_cost * CARD_VOLUME

        # 損益分岐に必要な追加販売数
        required_sales = investment / AVG_PROFIT

        # 必要なリピート率改善
        required_improvement = required_sales / REPEAT_BASE_CUSTOMERS

        breakeven[f"{card_cost}円/枚"] = {
            "投資額": investment,
            "必要追加販売数": int(np.ceil(required_sales)),
            "必要リピート率改善": f"+{required_improvement*100:.1f}%",
            "損益分岐リピート率": f"{(REPEAT_CURRENT_CVR + required_improvement)*100:.1f}%",
        }

    return breakeven


def calculate_scenario_comparison() -> pd.DataFrame:
    """
    シナリオ別の投資効果比較

    Returns:
        シナリオ比較DataFrame
    """
    scenarios = [
        {"name": "現状維持", "improvement": 0},
        {"name": "控えめ改善", "improvement": 0.03},  # +3%
        {"name": "目標達成", "improvement": 0.10},  # +10%（10%→20%）
        {"name": "楽観シナリオ", "improvement": 0.15},  # +15%
    ]

    results: List[Dict] = []

    for scenario in scenarios:
        improvement = scenario["improvement"]
        new_rate = REPEAT_CURRENT_CVR + improvement

        incremental_sales = REPEAT_BASE_CUSTOMERS * improvement
        incremental_revenue = incremental_sales * AVG_PRICE
        incremental_profit = incremental_sales * AVG_PROFIT

        for card_cost in [CARD_COST_LOW, CARD_COST_HIGH]:
            investment = card_cost * CARD_VOLUME
            net_profit = incremental_profit - investment
            roi = (net_profit / investment) * 100 if investment > 0 else 0

            results.append({
                "シナリオ": scenario["name"],
                "リピート率": f"{new_rate*100:.0f}%",
                "追加販売数": int(incremental_sales),
                "追加売上": int(incremental_revenue),
                "粗利増分": int(incremental_profit),
                "カード単価": f"{card_cost}円",
                "投資額": investment,
                "純利益": int(net_profit),
                "ROI": f"{roi:.0f}%",
                "評価": "◎" if roi >= 100 else ("○" if roi >= 0 else "×"),
            })

    return pd.DataFrame(results)


def plot_roi_curve() -> str:
    """
    ROI曲線グラフを作成

    Returns:
        保存先ファイルパス
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 左上: ROI曲線
    ax1 = axes[0, 0]
    improvements = np.arange(0, 0.16, 0.01) * 100  # %表示

    for card_cost, color in [(CARD_COST_LOW, "#2ca02c"), (CARD_COST_HIGH, "#d62728")]:
        investment = card_cost * CARD_VOLUME
        rois = []
        for imp in improvements / 100:
            incremental_sales = REPEAT_BASE_CUSTOMERS * imp
            incremental_profit = incremental_sales * AVG_PROFIT
            roi = (incremental_profit - investment) / investment * 100
            rois.append(roi)
        ax1.plot(improvements, rois, label=f"{card_cost}円/枚", color=color, linewidth=2)

    ax1.axhline(y=0, color="#333333", linestyle="--", linewidth=2)
    ax1.axhline(y=100, color="#666666", linestyle=":", alpha=0.5)
    ax1.fill_between(improvements, 0, 500, where=np.array(improvements) <= 5,
                     alpha=0.1, color="red")
    ax1.set_xlabel("リピート率改善（%ポイント）", fontsize=12)
    ax1.set_ylabel("ROI（%）", fontsize=12)
    ax1.set_title("同梱カード投資のROI曲線", fontsize=14, fontweight="bold")
    ax1.legend(loc="upper left")
    ax1.grid(alpha=0.3)
    ax1.set_ylim(-150, 500)

    # 損益分岐点の注記
    breakeven = find_breakeven()
    for i, (cost_label, be_info) in enumerate(breakeven.items()):
        imp_val = float(be_info["必要リピート率改善"].replace("+", "").replace("%", ""))
        ax1.annotate(
            f'{cost_label}\n損益分岐\n{be_info["必要リピート率改善"]}',
            xy=(imp_val, 0),
            xytext=(imp_val + 2, 50 + i * 80),
            arrowprops=dict(arrowstyle="->", color="#666666"),
            fontsize=9,
            ha="left"
        )

    # 右上: 累積粗利 vs 投資額
    ax2 = axes[0, 1]
    improvements_pct = np.arange(0, 16, 1)

    for card_cost, color in [(CARD_COST_LOW, "#2ca02c"), (CARD_COST_HIGH, "#d62728")]:
        investment = card_cost * CARD_VOLUME
        net_profits = []
        for imp in improvements_pct / 100:
            incremental_sales = REPEAT_BASE_CUSTOMERS * imp
            incremental_profit = incremental_sales * AVG_PROFIT
            net_profits.append((incremental_profit - investment) / 10000)  # 万円

        ax2.plot(improvements_pct, net_profits, "o-", label=f"{card_cost}円/枚",
                 color=color, linewidth=2, markersize=6)

    ax2.axhline(y=0, color="#333333", linestyle="--", linewidth=2)
    ax2.axvline(x=10, color="#4a90d9", linestyle=":", linewidth=2, alpha=0.7)
    ax2.text(10.5, ax2.get_ylim()[1] * 0.8, "目標\n(+10%)", fontsize=10, color="#4a90d9")

    ax2.set_xlabel("リピート率改善（%ポイント）", fontsize=12)
    ax2.set_ylabel("純利益（万円）", fontsize=12)
    ax2.set_title("リピート率改善と純利益の関係", fontsize=14, fontweight="bold")
    ax2.legend(loc="upper left")
    ax2.grid(alpha=0.3)

    # 左下: シナリオ比較（棒グラフ）
    ax3 = axes[1, 0]
    scenario_df = calculate_scenario_comparison()

    scenarios = ["現状維持", "控えめ改善", "目標達成", "楽観シナリオ"]
    x = np.arange(len(scenarios))
    width = 0.35

    low_profits = scenario_df[scenario_df["カード単価"] == "50円"]["純利益"].values / 10000
    high_profits = scenario_df[scenario_df["カード単価"] == "80円"]["純利益"].values / 10000

    bars1 = ax3.bar(x - width/2, low_profits, width, label="50円/枚", color="#2ca02c", alpha=0.8)
    bars2 = ax3.bar(x + width/2, high_profits, width, label="80円/枚", color="#d62728", alpha=0.8)

    ax3.axhline(y=0, color="#333333", linestyle="-", linewidth=1)
    ax3.set_xlabel("シナリオ", fontsize=12)
    ax3.set_ylabel("純利益（万円）", fontsize=12)
    ax3.set_title("シナリオ別・同梱カード投資の純利益", fontsize=14, fontweight="bold")
    ax3.set_xticks(x)
    ax3.set_xticklabels(scenarios)
    ax3.legend(loc="upper left")
    ax3.grid(axis="y", alpha=0.3)

    # 値ラベル
    for bar in bars1:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2, height + 0.5,
                 f"{height:.1f}", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2, height + 0.5,
                 f"{height:.1f}", ha="center", va="bottom", fontsize=9)

    # 右下: 投資回収期間（カード枚数別）
    ax4 = axes[1, 1]
    card_volumes = [500, 1000, 1500, 2000, 2500, 3000]

    for improvement, color, style in [(0.05, "#ff7f0e", "-"), (0.10, "#2ca02c", "-"),
                                       (0.15, "#1f77b4", "-")]:
        rois_50 = []
        rois_80 = []
        for volume in card_volumes:
            incremental_sales = REPEAT_BASE_CUSTOMERS * improvement
            incremental_profit = incremental_sales * AVG_PROFIT

            investment_50 = 50 * volume
            investment_80 = 80 * volume

            roi_50 = (incremental_profit - investment_50) / investment_50 * 100 if investment_50 > 0 else 0
            roi_80 = (incremental_profit - investment_80) / investment_80 * 100 if investment_80 > 0 else 0

            rois_50.append(roi_50)
            rois_80.append(roi_80)

        label_base = f"+{int(improvement*100)}%改善"
        ax4.plot(card_volumes, rois_50, "o-", color=color,
                 linewidth=2, label=f"{label_base}(50円)")
        ax4.plot(card_volumes, rois_80, "s:", color=color,
                 linewidth=2, alpha=0.6, label=f"{label_base}(80円)")

    ax4.axhline(y=0, color="#333333", linestyle="--", linewidth=2)
    ax4.axhline(y=100, color="#666666", linestyle=":", alpha=0.5)
    ax4.set_xlabel("同梱カード枚数", fontsize=12)
    ax4.set_ylabel("ROI（%）", fontsize=12)
    ax4.set_title("カード枚数別ROI（リピート率改善度別）", fontsize=14, fontweight="bold")
    ax4.legend(loc="upper right", fontsize=8)
    ax4.grid(alpha=0.3)

    plt.tight_layout()

    output_path = FIGURES_DIR / "04_card_roi_curve.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return str(output_path)


def run() -> Tuple[pd.DataFrame, Dict, str]:
    """
    シミュレーション項目4を実行

    Returns:
        (シナリオ比較DF, 損益分岐点情報, 保存先パス)のタプル
    """
    print("\n" + "=" * 60)
    print("シミュレーション項目4: 投資対効果（同梱カード）")
    print("=" * 60)

    print(f"\n【前提条件】")
    print(f"  既存顧客ベース: {REPEAT_BASE_CUSTOMERS:,}人")
    print(f"  現状リピート率: {REPEAT_CURRENT_CVR*100:.0f}%")
    print(f"  目標リピート率: {REPEAT_TARGET_CVR*100:.0f}%")
    print(f"  平均粗利/個: {AVG_PROFIT:,.0f}円")

    # 損益分岐点
    breakeven = find_breakeven()
    print("\n【損益分岐点分析】")
    for cost_label, be_info in breakeven.items():
        print(f"  {cost_label}:")
        print(f"    投資額: {be_info['投資額']:,}円")
        print(f"    必要追加販売数: {be_info['必要追加販売数']}個")
        print(f"    必要リピート率改善: {be_info['必要リピート率改善']}")
        print(f"    損益分岐リピート率: {be_info['損益分岐リピート率']}")

    # シナリオ比較
    scenario_df = calculate_scenario_comparison()
    print("\n【シナリオ別投資効果】")
    print(scenario_df.to_string(index=False))

    output_path = plot_roi_curve()
    print(f"\nグラフ保存先: {output_path}")

    # 判断基準
    print("\n【経営判断の目安】")
    print("  - リピート率+5%以上改善が見込める場合: 投資推奨")
    print("  - リピート率+10%改善で: ROI 200%超（50円/枚の場合）")
    print("  - 80円/枚でも+6%改善で損益分岐を超える")

    return scenario_df, breakeven, output_path


if __name__ == "__main__":
    run()
