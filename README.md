# ConvLSTMを用いた歩行者追跡システム

ConvLSTM（畳み込みLSTM）ニューラルネットワークを使用した多人数歩行者追跡システム。センサーヒートマップから歩行者座標を予測。

## 概要

このプロジェクトは、以下の機能を持つシナリオベースの歩行者追跡システムを実装しています：

- 複数センサーによる監視エリアでの歩行者移動シミュレーション
- ConvLSTMニューラルネットワークによる歩行者座標予測

## アーキテクチャ

### ConvLSTMモデル構成要素：
1. **畳み込みLSTM層**: 3層のConvLSTMで隠れ次元が段階的に増加（64→96→128）
2. **時間的注意機構**: 時間的特徴の重み付き結合
3. **マルチスケール空間エンコーダ**: 異なる空間スケール用の3つの並列ブランチ
4. **特徴融合**: マルチスケール特徴の結合
5. **座標ヘッド**: 各人物の(x,y)座標を予測
6. **信頼度ヘッド**: 検知信頼度を推定

### データパイプライン：
1. **シナリオ生成**: 現実的な歩行者移動シナリオの作成
2. **センサーシミュレーション**: センサーネットワークからの検知データ生成
3. **ヒートマップ作成**: センサーデータの空間表現への変換
4. **スライディングウィンドウ**: 学習用の逐次データ準備
5. **学習/検証**: データリークを防ぐシナリオベースのデータ分割

## システム要件

**実行環境**: Windows CPU  
**Pythonバージョン**: 3.8以上（Python 3.8-3.11で動作確認済み）

## インストール

1. このリポジトリをクローン：
```bash
git clone https://github.com/yourusername/pedestrian-tracking-convlstm.git
cd pedestrian-tracking-convlstm
```

2. 仮想環境の作成（推奨）：
```bash
python -m venv venv
venv\Scripts\activate  # Windows
```

3. 必要な依存関係をインストール：
```bash
pip install -r requirements.txt
```

## 依存関係

### 完全な要件：
バージョン仕様を含む完全なリストは`requirements.txt`を参照してください。

## 使用方法

### 基本的な使用方法：
```python
# 完全なパイプラインを実行
python new_senario_convlstm.py
```

## パフォーマンス注記

### Windows CPU最適化：
- CPU最適化されたPyTorch操作を使用
- メモリ管理のための設定可能なバッチサイズ
- 長時間実行操作の進捗監視

## トラブルシューティング

### よくある問題：

1. **メモリエラー**: `batch_size`または`num_samples`を減らしてください
2. **学習が遅い**: `window_size`を小さくするか、`num_epochs`を減らしてください。GPUがあれば切り替えてください
3. **インポートエラー**: `pip install -r requirements.txt`で全依存関係がインストールされていることを確認してください

## ライセンス

MIT License - 詳細はLICENSEファイルを参照

## 引用

この研究でこのコードを使用する場合は、以下のように引用してください：

```bibtex
@misc{pedestrian-tracking-convlstm,
  title={ConvLSTMを用いた歩行者追跡システム},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/pedestrian-tracking-convlstm}
}
```
