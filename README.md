# Krea2 AFM (experimental)

ComfyUI native Krea2 向けの独立した診断用カスタムノードです。
**現在は計画書の Task 0–1。Attention の編集機能は未実装です。**
Anima AFM や Krea2T Enhancer をインストールする必要はありません。

モデルを使った実機検証は今回の対象外です。CPU テストは API 契約・
状態分離・layout 検証の証拠であり、GPU 対応・高速化・画質改善を示しません。

## 対応の前提

- ComfyUI のネイティブ `comfy.ldm.krea2.model.SingleStreamDiT`。
- V3 ノード API、`OUTER_SAMPLE` / `DIFFUSION_MODEL` wrapper、`post_input`。
- ソース確認基準: ComfyUI `9d80841aa1990305cc7a280c5ea505f317efcc2d`。
  これは実機確認済み最低版ではありません。実行時にも必要 API を検査します。
- 通常の4D latent、時間軸が1の5D latent。reference / 複数フレームは対象外。
- 他の Attention override、post_input、QKV、Attention 出力パッチは競合エラー。

## インストールとテスト

このフォルダーを ComfyUI の `custom_nodes/AFM-for-Krea2-experimental` に配置し、
ComfyUI を再起動します。既存の ComfyUI 環境の PyTorch を使用します。
このパッケージのために PyTorch や GPU ドライバーを更新する必要はありません。

CPU テストは PyTorch と pytest が入った環境で次のように実行します。

```console
python -m pytest tests -q
python -m compileall -q afm nodes.py __init__.py
```

ノード、ログ、後日の実機確認については [検証手順](docs/validation.md) を参照。
[実装状況](docs/implementation-status.md) と [第三者資料](NOTICE.md) も同梱しています。
