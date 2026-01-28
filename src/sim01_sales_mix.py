"""
シミュレーション項目1: 販売ミックス最適化
5,000円帯と10,000円帯の組み合わせで10百万円達成パターンを算出
"""
from typing import List, Dict, Tuple

import pandas as pd
import matplotlib.pyplot as plt

from config import (
    TARGET_SALES,
    TIER_A_PRICE,
    TIER_A_PROFIT,
    TIER_B_PRICE,
    TIER_B_PROFIT,
    FIGURES_DIR,
)


def calculate_sales_mix_scenarios() -> pd.DataFrame:
    """
    販売ミックスシナリオを計算

    Returns:
        各シナリオの必要販売数と粗利を含むDataFrame
    """
    scenarios: List[Dict] = []

    # 5,000円帯の比率を0%〜100%で変化させる（10%刻み）
    for tier_a_ratio in range(0, 101, 10):
        tier_b_ratio = 100 - tier_a_ratio

        # 売上10百万円を達成するための販売個数を計算
        # tier_a_ratio% の売上が5,000円帯、tier_b_ratio% が10,000円帯
        if tier_a_ratio == 0:
            # 全て10,000円帯
            tier_a_units = 0
            tier_b_units = TARGET_SALES // TIER_B_PRICE
        elif tier_b_ratio == 0:
            # 全て5,000円帯
            tier_a_units = TARGET_SALES // TIER_A_PRICE
            tier_b_units = 0
        else:
            # 混合の場合、売上比率に基づいて計算
            tier_a_sales = TARGET_SALES * tier_a_ratio / 100
            tier_b_sales = TARGET_SALES * tier_b_ratio / 100
            tier_a_units = int(tier_a_sales / TIER_A_PRICE)
            tier_b_units = int(tier_b_sales / TIER_B_PRICE)

        total_units = tier_a_units + tier_b_units
        total_sales = tier_a_units * TIER_A_PRICE + tier_b_units * TIER_B_PRICE
        gross_profit = tier_a_units * TIER_A_PROFIT + tier_b_units * TIER_B_PROFIT

        scenarios.append({
            "シナリオ": f"A{tier_a_ratio}:B{tier_b_ratio}",
            "5000円帯比率": tier_a_ratio,
            "10000円帯比率": tier_b_ratio,
            "5000円帯個数": tier_a_units,
            "10000円帯個数": tier_b_units,
            "合計個数": total_units,
            "売上": total_sales,
            "粗利": gross_profit,
            "粗利率": gross_profit / total_sales if total_sales > 0 else 0,
        })

    return pd.DataFrame(scenarios)


def plot_sales_mix(df: pd.DataFrame) -> str:
    """
    販売ミックスシナリオのグラフを作成

    Args:
        df: シナリオデータ

    Returns:
        保存先ファイルパス
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 左グラフ: 必要販売個数（積み上げ棒グラフ）
    ax1 = axes[0]
    x = range(len(df))
    ax1.bar(x, df["5000円帯個数"], label="5,000円帯", color="#4a90d9", alpha=0.8)
    ax1.bar(x, df["10000円帯個数"], bottom=df["5000円帯個数"],
            label="10,000円帯", color="#e07b54", alpha=0.8)
    ax1.set_xlabel("販売ミックス比率（A:B）", fontsize=12)
    ax1.set_ylabel("必要販売個数", fontsize=12)
    ax1.set_title("販売ミックス別・必要販売個数", fontsize=14, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(df["シナリオ"], rotation=45, ha="right")
    ax1.legend(loc="upper right")
    ax1.grid(axis="y", alpha=0.3)

    # 右グラフ: 粗利額
    ax2 = axes[1]
    colors = ["#2ca02c" if p >= 1_500_000 else "#d62728" for p in df["粗利"]]
    bars = ax2.bar(x, df["粗利"] / 10000, color=colors, alpha=0.8)
    ax2.axhline(y=150, color="#333333", linestyle="--", linewidth=2,
                label="目標粗利(150万円)")
    ax2.set_xlabel("販売ミックス比率（A:B）", fontsize=12)
    ax2.set_ylabel("粗利（万円）", fontsize=12)
    ax2.set_title("販売ミックス別・粗利額", fontsize=14, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(df["シナリオ"], rotation=45, ha="right")
    ax2.legend(loc="upper right")
    ax2.grid(axis="y", alpha=0.3)

    # 棒グラフの上に値を表示
    for bar, val in zip(bars, df["粗利"]):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                 f"{val/10000:.0f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()

    output_path = FIGURES_DIR / "01_sales_mix_scenarios.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return str(output_path)


def run() -> Tuple[pd.DataFrame, str]:
    """
    シミュレーション項目1を実行

    Returns:
        (シナリオデータ, 保存先パス)のタプル
    """
    print("=" * 60)
    print("シミュレーション項目1: 販売ミックス最適化")
    print("=" * 60)

    df = calculate_sales_mix_scenarios()
    print("\n【販売ミックスシナリオ一覧】")
    print(df.to_string(index=False))

    output_path = plot_sales_mix(df)
    print(f"\nグラフ保存先: {output_path}")

    # 最適シナリオの特定
    best = df.loc[df["粗利"].idxmax()]
    print(f"\n【粗利最大シナリオ】")
    print(f"  シナリオ: {best['シナリオ']}")
    print(f"  粗利: {best['粗利']:,.0f}円")
    print(f"  必要販売個数: {best['合計個数']:,.0f}個")

    return df, output_path


if __name__ == "__main__":
    run()
