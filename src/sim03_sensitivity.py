"""
シミュレーション項目3: 感度分析
転換率や価格帯比率の変動が粗利に与える影響を分析
"""
from typing import Tuple, List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import (
    TARGET_SALES,
    TIER_A_PRICE,
    TIER_A_PROFIT,
    TIER_B_PRICE,
    TIER_B_PROFIT,
    CHANNELS,
    FIGURES_DIR,
)


AVG_PRICE = (TIER_A_PRICE + TIER_B_PRICE) / 2
AVG_PROFIT = (TIER_A_PROFIT + TIER_B_PROFIT) / 2


def cvr_sensitivity_analysis() -> pd.DataFrame:
    """
    転換率感度分析を実行

    転換率が±0.1%〜±0.5%変動した場合の期待販売数・粗利への影響

    Returns:
        感度分析結果DataFrame
    """
    results: List[Dict] = []

    # 変動幅
    deltas = [-0.005, -0.003, -0.001, 0, 0.001, 0.003, 0.005]
    delta_labels = ["-0.5%", "-0.3%", "-0.1%", "基準", "+0.1%", "+0.3%", "+0.5%"]

    for ch_name, ch_info in CHANNELS.items():
        base_cvr = ch_info["target_cvr"]
        traffic = ch_info["annual_traffic"]

        for delta, label in zip(deltas, delta_labels):
            new_cvr = max(0.0001, base_cvr + delta)  # 最小0.01%
            expected_sales = int(traffic * new_cvr)
            expected_revenue = expected_sales * AVG_PRICE
            expected_profit = expected_sales * AVG_PROFIT

            results.append({
                "チャネル": ch_name,
                "変動": label,
                "調整後CVR": f"{new_cvr*100:.2f}%",
                "期待販売数": expected_sales,
                "期待売上": expected_revenue,
                "期待粗利": expected_profit,
            })

    return pd.DataFrame(results)


def price_mix_sensitivity() -> pd.DataFrame:
    """
    価格帯比率の感度分析

    Returns:
        感度分析結果DataFrame
    """
    results: List[Dict] = []

    # 基準: 50:50
    base_ratio = 50

    for tier_a_ratio in range(0, 101, 10):
        tier_b_ratio = 100 - tier_a_ratio

        tier_a_sales = TARGET_SALES * tier_a_ratio / 100
        tier_b_sales = TARGET_SALES * tier_b_ratio / 100

        tier_a_units = int(tier_a_sales / TIER_A_PRICE)
        tier_b_units = int(tier_b_sales / TIER_B_PRICE)

        total_units = tier_a_units + tier_b_units
        gross_profit = tier_a_units * TIER_A_PROFIT + tier_b_units * TIER_B_PROFIT

        # 基準（50:50）からの変化
        base_profit = int(TARGET_SALES * 0.5 / TIER_A_PRICE) * TIER_A_PROFIT + \
                      int(TARGET_SALES * 0.5 / TIER_B_PRICE) * TIER_B_PROFIT
        profit_change = gross_profit - base_profit
        profit_change_pct = (profit_change / base_profit) * 100 if base_profit > 0 else 0

        results.append({
            "5000円帯比率": f"{tier_a_ratio}%",
            "10000円帯比率": f"{tier_b_ratio}%",
            "合計販売個数": total_units,
            "粗利": gross_profit,
            "基準比変化": profit_change,
            "変化率": f"{profit_change_pct:+.1f}%",
        })

    return pd.DataFrame(results)


def plot_sensitivity() -> str:
    """
    感度分析グラフ（トルネードチャート風）を作成

    Returns:
        保存先ファイルパス
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 左上: チャネル別CVR感度（線グラフ）
    ax1 = axes[0, 0]
    deltas = np.array([-0.5, -0.3, -0.1, 0, 0.1, 0.3, 0.5])
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for i, (ch_name, ch_info) in enumerate(CHANNELS.items()):
        base_cvr = ch_info["target_cvr"]
        traffic = ch_info["annual_traffic"]

        profits = []
        for d in deltas / 100:
            new_cvr = max(0.0001, base_cvr + d)
            expected_sales = traffic * new_cvr
            expected_profit = expected_sales * AVG_PROFIT
            profits.append(expected_profit / 10000)  # 万円

        ax1.plot(deltas, profits, marker="o", label=ch_name, color=colors[i], linewidth=2)

    ax1.axvline(x=0, color="#666666", linestyle="--", alpha=0.5)
    ax1.set_xlabel("CVR変動（%ポイント）", fontsize=12)
    ax1.set_ylabel("期待粗利（万円）", fontsize=12)
    ax1.set_title("転換率変動の影響（チャネル別）", fontsize=14, fontweight="bold")
    ax1.legend(loc="upper left")
    ax1.grid(alpha=0.3)

    # 右上: トルネードチャート（CVR +0.5%の影響度比較）
    ax2 = axes[0, 1]
    impact_data = []
    for ch_name, ch_info in CHANNELS.items():
        base_cvr = ch_info["target_cvr"]
        traffic = ch_info["annual_traffic"]

        base_profit = traffic * base_cvr * AVG_PROFIT
        plus_profit = traffic * (base_cvr + 0.005) * AVG_PROFIT
        impact = plus_profit - base_profit
        impact_data.append({"チャネル": ch_name, "影響額": impact / 10000})

    impact_df = pd.DataFrame(impact_data).sort_values("影響額", ascending=True)
    y_pos = range(len(impact_df))
    bars = ax2.barh(y_pos, impact_df["影響額"],
                    color=["#2ca02c" if v > 0 else "#d62728" for v in impact_df["影響額"]])
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(impact_df["チャネル"])
    ax2.set_xlabel("粗利増加額（万円）", fontsize=12)
    ax2.set_title("CVR +0.5%改善時の粗利インパクト", fontsize=14, fontweight="bold")
    ax2.grid(axis="x", alpha=0.3)

    for bar in bars:
        width = bar.get_width()
        ax2.text(width + 0.5, bar.get_y() + bar.get_height()/2,
                 f"{width:.1f}", ha="left", va="center", fontsize=10)

    # 左下: 価格帯比率の感度
    ax3 = axes[1, 0]
    mix_df = price_mix_sensitivity()
    tier_a_ratios = [int(r.replace("%", "")) for r in mix_df["5000円帯比率"]]
    profits = mix_df["粗利"].values / 10000
    units = mix_df["合計販売個数"].values

    ax3_twin = ax3.twinx()
    line1, = ax3.plot(tier_a_ratios, profits, "o-", color="#2ca02c",
                      linewidth=2, markersize=8, label="粗利（万円）")
    line2, = ax3_twin.plot(tier_a_ratios, units, "s--", color="#d62728",
                           linewidth=2, markersize=8, label="販売個数")

    ax3.axhline(y=150, color="#333333", linestyle=":", linewidth=2, alpha=0.7)
    ax3.text(5, 152, "目標粗利150万円", fontsize=10)

    ax3.set_xlabel("5,000円帯比率（%）", fontsize=12)
    ax3.set_ylabel("粗利（万円）", fontsize=12, color="#2ca02c")
    ax3_twin.set_ylabel("販売個数", fontsize=12, color="#d62728")
    ax3.set_title("価格帯ミックスと粗利・販売個数の関係", fontsize=14, fontweight="bold")
    ax3.grid(alpha=0.3)
    ax3.legend(handles=[line1, line2], loc="upper right")

    # 右下: 全チャネル合計の感度曲線
    ax4 = axes[1, 1]
    cvr_multipliers = np.linspace(0.5, 3.0, 20)  # 基準CVRの0.5倍〜3倍

    total_profits = []
    for mult in cvr_multipliers:
        total = 0
        for ch_info in CHANNELS.values():
            new_cvr = ch_info["target_cvr"] * mult
            total += ch_info["annual_traffic"] * new_cvr * AVG_PROFIT
        total_profits.append(total / 10000)

    ax4.fill_between(cvr_multipliers, total_profits, alpha=0.3, color="#4a90d9")
    ax4.plot(cvr_multipliers, total_profits, color="#4a90d9", linewidth=2)
    ax4.axhline(y=150, color="#333333", linestyle="--", linewidth=2)
    ax4.axvline(x=1.0, color="#666666", linestyle=":", alpha=0.5)

    ax4.set_xlabel("目標CVRに対する倍率", fontsize=12)
    ax4.set_ylabel("全チャネル合計粗利（万円）", fontsize=12)
    ax4.set_title("CVR改善度合いと全体粗利の関係", fontsize=14, fontweight="bold")
    ax4.grid(alpha=0.3)
    ax4.text(1.02, total_profits[9] + 5, f"目標CVR時\n{total_profits[9]:.0f}万円", fontsize=10)

    plt.tight_layout()

    output_path = FIGURES_DIR / "03_cvr_sensitivity.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return str(output_path)


def run() -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """
    シミュレーション項目3を実行

    Returns:
        (CVR感度DF, 価格帯感度DF, 保存先パス)のタプル
    """
    print("\n" + "=" * 60)
    print("シミュレーション項目3: 感度分析")
    print("=" * 60)

    cvr_df = cvr_sensitivity_analysis()
    print("\n【CVR感度分析（抜粋: 旅客メルマガ）】")
    sample = cvr_df[cvr_df["チャネル"] == "旅客メルマガ"]
    print(sample.to_string(index=False))

    mix_df = price_mix_sensitivity()
    print("\n【価格帯比率感度分析】")
    print(mix_df.to_string(index=False))

    output_path = plot_sensitivity()
    print(f"\nグラフ保存先: {output_path}")

    # 重要な知見
    print("\n【感度分析の知見】")

    # CVR影響度ランキング
    impact_ranking = []
    for ch_name, ch_info in CHANNELS.items():
        impact = ch_info["annual_traffic"] * 0.005 * AVG_PROFIT  # CVR +0.5%の影響
        impact_ranking.append((ch_name, impact))
    impact_ranking.sort(key=lambda x: x[1], reverse=True)

    print("  CVR +0.5%改善時の粗利インパクト（大→小）:")
    for rank, (ch, imp) in enumerate(impact_ranking, 1):
        print(f"    {rank}. {ch}: +{imp/10000:.1f}万円")

    return cvr_df, mix_df, output_path


if __name__ == "__main__":
    run()
