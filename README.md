# EEZO収益シミュレーション

## 概要

北海道食材専門商社のEC事業「EEZO」の収益モデルをシミュレーションするためのリポジトリです。

**Claude Code on the Web** での利用を想定しています。

## 目的

- 年間売上10百万円達成に必要な条件の可視化
- チャネル別の転換率改善効果の試算
- 同梱カード投資の費用対効果検証

## 使い方

### 1. GitHubにプッシュ

```bash
git init
git add .
git commit -m "Initial setup"
git remote add origin https://github.com/[your-username]/eezo-revenue-sim.git
git push -u origin main
```

### 2. Claude Code on the Web でリポジトリ選択

https://claude.ai/code にアクセスし、このリポジトリを選択

### 3. 指示を入力

```
CLAUDE.mdを読んで、シミュレーション項目1〜4を順次実行して
```

## ディレクトリ構成

```
eezo-revenue-sim/
├── CLAUDE.md                 # プロジェクト指示書（最重要）
├── README.md                 # このファイル
├── requirements.txt          # Python依存パッケージ
├── .gitignore
│
├── .claude/
│   └── settings.json         # 自動セットアップ用フック
│
├── data/
│   └── raw/
│       └── assumptions.csv   # 前提条件データ
│
├── experiments/
│   └── exp001_baseline/
│       └── outputs/          # 出力先
│
└── src/
    └── utils/                # 再利用コード（必要に応じて）
```

## 前提条件

| 項目 | 値 |
|------|-----|
| 目標売上 | 10,000,000円/年 |
| 粗利率 | 15% |
| 価格帯 | 5,000円 / 10,000円 |

## 出力物

- シミュレーションレポート（Markdown）
- 可視化グラフ（PNG）
  - 販売ミックスシナリオ
  - チャネル別必要流入数
  - 転換率感度分析
  - 同梱カードROI曲線

## 技術スタック

- Python 3.x
- pandas, numpy, matplotlib, seaborn
- japanize-matplotlib（日本語フォント対応）

---

*新日本海商事 EEZO事業戦略検討用*
