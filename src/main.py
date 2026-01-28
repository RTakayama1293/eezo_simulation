"""
EEZO収益シミュレーション - メイン実行スクリプト
シミュレーション項目1〜4を順次実行し、レポートを生成する
"""
import sys
from pathlib import Path
from datetime import datetime

# srcディレクトリをパスに追加
sys.path.insert(0, str(Path(__file__).parent))

from config import OUTPUT_DIR, FIGURES_DIR, TARGET_SALES, TARGET_GROSS_PROFIT
import sim01_sales_mix
import sim02_channel_scenario
import sim03_sensitivity
import sim04_card_roi


def generate_report(
    sales_mix_df,
    channel_gap_df,
    cvr_sensitivity_df,
    price_mix_df,
    card_scenario_df,
    card_breakeven
) -> str:
    """
    シミュレーションレポートを生成

    Returns:
        レポートファイルパス
    """
    report_path = OUTPUT_DIR / "simulation_report.md"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# EEZO収益シミュレーションレポート\n\n")
        f.write(f"**生成日時**: {datetime.now().strftime('%Y年%m月%d日 %H:%M')}\n\n")
        f.write(f"**目標**: 年間売上 {TARGET_SALES/10000:,.0f}万円、粗利 {TARGET_GROSS_PROFIT/10000:,.0f}万円\n\n")
        f.write("---\n\n")

        # エグゼクティブサマリー
        f.write("## エグゼクティブサマリー\n\n")
        f.write("### 主要な発見事項\n\n")

        # 販売ミックス最適解
        best_mix = sales_mix_df.loc[sales_mix_df["粗利"].idxmax()]
        f.write(f"1. **販売ミックス最適化**: 10,000円帯商品の比率を高めるほど粗利が増加。"
                f"全て10,000円帯（{best_mix['シナリオ']}）で粗利 **{best_mix['粗利']/10000:.0f}万円**（目標達成）\n\n")

        # チャネル評価
        total_target_sales = sum(
            float(row["目標期待販売数"])
            for _, row in channel_gap_df.iterrows()
        )
        target_units = int(TARGET_SALES / 7500)
        f.write(f"2. **チャネル別達成度**: 目標CVR達成時の全チャネル合計期待販売数は"
                f" **{int(total_target_sales):,}個**（目標{target_units:,}個の{total_target_sales/target_units*100:.0f}%）\n\n")

        # 感度分析の知見
        f.write("3. **感度分析**: CVR改善インパクトが最も大きいのは**旅客メルマガ**"
                "（流入数が多いため0.5%改善で大きな効果）\n\n")

        # 同梱カードROI
        be_50 = card_breakeven["50円/枚"]
        f.write(f"4. **同梱カード投資**: 50円/枚の場合、リピート率"
                f"**{be_50['必要リピート率改善']}**改善で損益分岐。"
                "目標達成（+10%）時のROIは200%超\n\n")

        # 達成確率評価
        f.write("### 達成確率評価\n\n")
        f.write("| シナリオ | 達成確率 | 根拠 |\n")
        f.write("|---------|---------|------|\n")
        f.write("| 現状維持 | **低** | CVR 0.04%では目標の1%未満 |\n")
        f.write("| CVR改善（目標値） | **中** | 全チャネル合計でも目標の約30%程度 |\n")
        f.write("| CVR改善+商品力強化 | **中〜高** | 単価アップと複数チャネル組み合わせが必要 |\n")
        f.write("| 抜本的施策 | **要検討** | 新規チャネル開拓、商品数拡大が必要 |\n\n")

        f.write("---\n\n")

        # 詳細分析
        f.write("## 1. 販売ミックス最適化\n\n")
        f.write("5,000円帯と10,000円帯の組み合わせで年間売上1,000万円を達成するパターンを分析。\n\n")
        f.write("### シナリオ一覧\n\n")
        f.write("| シナリオ | 5,000円帯 | 10,000円帯 | 合計個数 | 粗利 |\n")
        f.write("|---------|----------|-----------|---------|------|\n")
        for _, row in sales_mix_df.iterrows():
            f.write(f"| {row['シナリオ']} | {row['5000円帯個数']:,}個 | "
                    f"{row['10000円帯個数']:,}個 | {row['合計個数']:,}個 | "
                    f"{row['粗利']/10000:.0f}万円 |\n")

        f.write("\n### 考察\n\n")
        f.write("- 10,000円帯の比率を高めると、販売個数は減るが**粗利率が向上**\n")
        f.write("- 全て10,000円帯（A0:B100）の場合、1,000個販売で目標達成・粗利150万円\n")
        f.write("- 5,000円帯のみでは2,000個販売が必要（物流・在庫負担大）\n\n")

        f.write("![販売ミックス](./figures/01_sales_mix_scenarios.png)\n\n")

        f.write("---\n\n")

        # チャネル別シナリオ
        f.write("## 2. チャネル別達成シナリオ\n\n")
        f.write("各チャネルの現状CVRと目標CVRで必要流入数・期待販売数を分析。\n\n")
        f.write("### ギャップ分析\n\n")
        f.write("| チャネル | 年間流入 | 現状CVR | 目標CVR | 現状販売 | 目標販売 | 評価 |\n")
        f.write("|---------|---------|---------|---------|---------|---------|------|\n")
        for _, row in channel_gap_df.iterrows():
            f.write(f"| {row['チャネル']} | {row['年間流入見込']:,} | {row['現状CVR']} | "
                    f"{row['目標CVR']} | {row['現状期待販売数']:,}個 | "
                    f"{row['目標期待販売数']:,}個 | {row['評価']} |\n")

        f.write("\n### 考察\n\n")
        f.write("- **旅客メルマガ**: 流入数は多いがCVRが極めて低い。0.5%まで改善できれば主力チャネルに\n")
        f.write("- **O2O**: CVR 3%は現実的。船内体験との連携が鍵\n")
        f.write("- **同梱リピート**: 少数だが高CVR。既存顧客のLTV最大化に有効\n")
        f.write("- **SNS**: 中間的なポテンシャル。コンテンツ力次第\n\n")

        f.write("![チャネル分析](./figures/02_channel_traffic_matrix.png)\n\n")

        f.write("---\n\n")

        # 感度分析
        f.write("## 3. 感度分析\n\n")
        f.write("### CVR変動の影響\n\n")
        f.write("転換率が±0.1%〜±0.5%変動した場合の期待粗利への影響を分析。\n\n")

        f.write("**CVR +0.5%改善時のインパクト（チャネル別）**:\n\n")
        channels_info = [
            ("旅客メルマガ", 400000, 0.005),
            ("O2O", 50000, 0.03),
            ("SNS", 100000, 0.01),
            ("同梱リピート", 2000, 0.20),
        ]
        avg_profit = 1125  # 平均粗利
        for ch, traffic, _ in channels_info:
            impact = traffic * 0.005 * avg_profit / 10000
            f.write(f"- {ch}: +{impact:.1f}万円\n")

        f.write("\n### 価格帯比率の影響\n\n")
        f.write("| 5,000円帯比率 | 10,000円帯比率 | 粗利 | 基準比変化 |\n")
        f.write("|-------------|--------------|------|----------|\n")
        for _, row in price_mix_df.iterrows():
            f.write(f"| {row['5000円帯比率']} | {row['10000円帯比率']} | "
                    f"{row['粗利']/10000:.0f}万円 | {row['変化率']} |\n")

        f.write("\n![感度分析](./figures/03_cvr_sensitivity.png)\n\n")

        f.write("---\n\n")

        # 同梱カードROI
        f.write("## 4. 投資対効果（同梱カード）\n\n")
        f.write("### 前提条件\n\n")
        f.write("- 既存顧客ベース: 2,000人\n")
        f.write("- 現状リピート率: 10%\n")
        f.write("- 目標リピート率: 20%（+10%改善）\n")
        f.write("- カード単価: 50〜80円/枚\n")
        f.write("- 発注枚数: 2,000枚\n\n")

        f.write("### 損益分岐点\n\n")
        f.write("| カード単価 | 投資額 | 必要追加販売数 | 必要リピート率改善 |\n")
        f.write("|-----------|-------|--------------|------------------|\n")
        for cost_label, be_info in card_breakeven.items():
            f.write(f"| {cost_label} | {be_info['投資額']:,}円 | "
                    f"{be_info['必要追加販売数']}個 | {be_info['必要リピート率改善']} |\n")

        f.write("\n### シナリオ別ROI\n\n")
        f.write("| シナリオ | リピート率 | 追加販売 | 粗利増分 | ROI(50円) | ROI(80円) | 評価 |\n")
        f.write("|---------|----------|---------|---------|----------|----------|------|\n")

        scenarios = ["現状維持", "控えめ改善", "目標達成", "楽観シナリオ"]
        for scenario in scenarios:
            rows = card_scenario_df[card_scenario_df["シナリオ"] == scenario]
            row_50 = rows[rows["カード単価"] == "50円"].iloc[0]
            row_80 = rows[rows["カード単価"] == "80円"].iloc[0]
            f.write(f"| {scenario} | {row_50['リピート率']} | {row_50['追加販売数']}個 | "
                    f"{row_50['粗利増分']/10000:.1f}万円 | {row_50['ROI']} | {row_80['ROI']} | "
                    f"{row_50['評価']} |\n")

        f.write("\n### 考察\n\n")
        f.write("- **投資推奨条件**: リピート率+5%以上の改善が見込める場合\n")
        f.write("- **目標達成時（+10%）**: ROI 200%超と高い投資効率\n")
        f.write("- **リスク**: 改善効果が+3%未満の場合は赤字リスクあり\n\n")

        f.write("![同梱カードROI](./figures/04_card_roi_curve.png)\n\n")

        f.write("---\n\n")

        # 推奨アクション
        f.write("## 推奨アクション\n\n")
        f.write("### 優先度：高\n\n")
        f.write("1. **10,000円帯商品の拡充** - 粗利率向上の最大レバー\n")
        f.write("2. **旅客メルマガのCVR改善** - 0.04%→0.5%で大きなインパクト\n")
        f.write("3. **O2Oの強化** - 船内体験との連携でCVR 3%を目指す\n\n")

        f.write("### 優先度：中\n\n")
        f.write("4. **同梱カード導入** - 目標リピート率達成でROI 200%超\n")
        f.write("5. **SNSコンテンツ強化** - 商品ストーリーの発信\n\n")

        f.write("### 優先度：低（中長期）\n\n")
        f.write("6. **新規チャネル開拓** - 現状チャネルだけでは目標達成困難\n")
        f.write("7. **商品数拡大** - 35商品→100商品規模への拡大検討\n\n")

        f.write("---\n\n")
        f.write("*本レポートはシミュレーション結果に基づく参考値です。*\n")

    return str(report_path)


def main():
    """
    メイン実行関数
    """
    print("=" * 70)
    print("EEZO収益シミュレーション 実行開始")
    print(f"実行日時: {datetime.now().strftime('%Y年%m月%d日 %H:%M')}")
    print("=" * 70)

    # ディレクトリ確認
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # シミュレーション項目1: 販売ミックス最適化
    sales_mix_df, _ = sim01_sales_mix.run()

    # シミュレーション項目2: チャネル別達成シナリオ
    channel_gap_df, _ = sim02_channel_scenario.run()

    # シミュレーション項目3: 感度分析
    cvr_df, price_mix_df, _ = sim03_sensitivity.run()

    # シミュレーション項目4: 投資対効果
    card_scenario_df, card_breakeven, _ = sim04_card_roi.run()

    # レポート生成
    print("\n" + "=" * 60)
    print("レポート生成")
    print("=" * 60)

    report_path = generate_report(
        sales_mix_df,
        channel_gap_df,
        cvr_df,
        price_mix_df,
        card_scenario_df,
        card_breakeven
    )
    print(f"レポート保存先: {report_path}")

    print("\n" + "=" * 70)
    print("全シミュレーション完了")
    print("=" * 70)

    print("\n【出力ファイル一覧】")
    print(f"  グラフ:")
    for fig in sorted(FIGURES_DIR.glob("*.png")):
        print(f"    - {fig.name}")
    print(f"  レポート:")
    print(f"    - simulation_report.md")


if __name__ == "__main__":
    main()
