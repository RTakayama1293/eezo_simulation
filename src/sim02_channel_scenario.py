"""
シミュレーション項目2: チャネル別達成シナリオ
各チャネルの転換率別に必要流入数を算出し、ギャップ分析を行う
"""
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from config import (
    TARGET_SALES,
    TIER_A_PRICE,
    TIER_B_PRICE,
    CHANNELS,
    FIGURES_DIR,
)


# 平均販売単価（5,000円と10,000円の中間を想定）
AVG_PRICE = (TIER_A_PRICE + TIER_B_PRICE) / 2  # 7,500円
TARGET_UNITS = int(TARGET_SALES / AVG_PRICE)  # 約1,333個


def calculate_required_traffic(cvr: float, target_units: int) -> int:
    """
    必要流入数を計算

    Args:
        cvr: 転換率
        target_units: 目標販売個数

    Returns:
        必要流入数
    """
    if cvr <= 0:
        return float("inf")
    return int(target_units / cvr)


def build_channel_matrix() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    チャネル別・転換率別の必要流入数マトリックスを作成

    Returns:
        (マトリックスDF, ギャップ分析DF)のタプル
    """
    # 転換率の範囲（0.1%〜5%を0.5%刻み + 現状値・目標値）
    cvr_values = [0.001, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05]
    cvr_labels = [f"{v*100:.1f}%" for v in cvr_values]

    matrix_data = {}
    for cvr, label in zip(cvr_values, cvr_labels):
        matrix_data[label] = [calculate_required_traffic(cvr, TARGET_UNITS)]

    matrix_df = pd.DataFrame(matrix_data, index=["必要流入数"])

    # ギャップ分析
    gap_data = []
    for ch_name, ch_info in CHANNELS.items():
        current_cvr = ch_info["current_cvr"]
        target_cvr = ch_info["target_cvr"]
        annual_traffic = ch_info["annual_traffic"]

        # 現状CVRでの必要流入数
        required_current = calculate_required_traffic(current_cvr, TARGET_UNITS)
        # 目標CVRでの必要流入数
        required_target = calculate_required_traffic(target_cvr, TARGET_UNITS)

        # 現状での期待販売数
        expected_sales_current = int(annual_traffic * current_cvr)
        # 目標CVRでの期待販売数
        expected_sales_target = int(annual_traffic * target_cvr)

        # ギャップ（目標達成に必要な追加流入）
        gap_current = max(0, required_current - annual_traffic)
        gap_target = max(0, required_target - annual_traffic)

        # 達成率
        achievement_current = min(100, annual_traffic / required_current * 100) if required_current > 0 else 0
        achievement_target = min(100, annual_traffic / required_target * 100) if required_target > 0 else 0

        gap_data.append({
            "チャネル": ch_name,
            "年間流入見込": annual_traffic,
            "現状CVR": f"{current_cvr*100:.2f}%",
            "目標CVR": f"{target_cvr*100:.1f}%",
            "現状期待販売数": expected_sales_current,
            "目標期待販売数": expected_sales_target,
            "現状達成率": f"{achievement_current:.1f}%",
            "目標達成率": f"{achievement_target:.1f}%",
            "評価": "◎" if achievement_target >= 50 else ("○" if achievement_target >= 20 else "△"),
        })

    gap_df = pd.DataFrame(gap_data)
    return matrix_df, gap_df


def plot_channel_matrix() -> str:
    """
    チャネル別・必要流入数のヒートマップを作成

    Returns:
        保存先ファイルパス
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # 左: 各チャネル×CVRの必要流入数ヒートマップ
    cvr_values = [0.001, 0.005, 0.01, 0.02, 0.03, 0.05]
    cvr_labels = [f"{v*100:.1f}%" for v in cvr_values]

    heatmap_data = []
    for ch_name in CHANNELS.keys():
        row = []
        for cvr in cvr_values:
            required = calculate_required_traffic(cvr, TARGET_UNITS)
            row.append(required / 10000)  # 万単位
        heatmap_data.append(row)

    heatmap_df = pd.DataFrame(
        heatmap_data,
        index=list(CHANNELS.keys()),
        columns=cvr_labels
    )

    ax1 = axes[0]
    sns.heatmap(
        heatmap_df,
        annot=True,
        fmt=".1f",
        cmap="YlOrRd_r",
        ax=ax1,
        cbar_kws={"label": "必要流入数（万）"}
    )
    ax1.set_title("CVR別・目標達成に必要な流入数（万人/クリック）",
                  fontsize=14, fontweight="bold")
    ax1.set_xlabel("転換率（CVR）", fontsize=12)
    ax1.set_ylabel("チャネル", fontsize=12)

    # 右: 各チャネルの現状vs期待販売数
    ax2 = axes[1]

    channels = list(CHANNELS.keys())
    x = np.arange(len(channels))
    width = 0.35

    current_sales = []
    target_sales = []
    for ch_info in CHANNELS.values():
        current_sales.append(ch_info["annual_traffic"] * ch_info["current_cvr"])
        target_sales.append(ch_info["annual_traffic"] * ch_info["target_cvr"])

    bars1 = ax2.bar(x - width/2, current_sales, width, label="現状CVR",
                    color="#d62728", alpha=0.7)
    bars2 = ax2.bar(x + width/2, target_sales, width, label="目標CVR",
                    color="#2ca02c", alpha=0.7)

    ax2.axhline(y=TARGET_UNITS, color="#333333", linestyle="--", linewidth=2,
                label=f"目標販売数({TARGET_UNITS:,}個)")
    ax2.set_xlabel("チャネル", fontsize=12)
    ax2.set_ylabel("期待販売数（個）", fontsize=12)
    ax2.set_title("チャネル別・期待販売数（流入見込×CVR）",
                  fontsize=14, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(channels, rotation=15, ha="right")
    ax2.legend(loc="upper right")
    ax2.grid(axis="y", alpha=0.3)

    # 値ラベル
    for bar in bars1:
        height = bar.get_height()
        if height > 0:
            ax2.text(bar.get_x() + bar.get_width()/2, height + 20,
                     f"{int(height)}", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        if height > 0:
            ax2.text(bar.get_x() + bar.get_width()/2, height + 20,
                     f"{int(height)}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()

    output_path = FIGURES_DIR / "02_channel_traffic_matrix.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return str(output_path)


def run() -> Tuple[pd.DataFrame, str]:
    """
    シミュレーション項目2を実行

    Returns:
        (ギャップ分析DF, 保存先パス)のタプル
    """
    print("\n" + "=" * 60)
    print("シミュレーション項目2: チャネル別達成シナリオ")
    print("=" * 60)

    print(f"\n目標販売個数: {TARGET_UNITS:,}個（平均単価{AVG_PRICE:,.0f}円で計算）")

    _, gap_df = build_channel_matrix()
    print("\n【チャネル別ギャップ分析】")
    print(gap_df.to_string(index=False))

    output_path = plot_channel_matrix()
    print(f"\nグラフ保存先: {output_path}")

    # 合計期待販売数
    total_current = sum(
        ch["annual_traffic"] * ch["current_cvr"]
        for ch in CHANNELS.values()
    )
    total_target = sum(
        ch["annual_traffic"] * ch["target_cvr"]
        for ch in CHANNELS.values()
    )

    print(f"\n【全チャネル合計】")
    print(f"  現状CVRでの期待販売数: {int(total_current):,}個 "
          f"（目標比: {total_current/TARGET_UNITS*100:.1f}%）")
    print(f"  目標CVRでの期待販売数: {int(total_target):,}個 "
          f"（目標比: {total_target/TARGET_UNITS*100:.1f}%）")

    return gap_df, output_path


if __name__ == "__main__":
    run()
