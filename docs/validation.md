# 検証範囲

この版は Task 0–1 の診断版です。今回、実モデルの推論は実施しません。

| 項目 | 根拠 / 状態 |
| --- | --- |
| Python / PyTorch | ローカル Python 3.10.11 / PyTorch 2.10.0+cpu |
| Krea2 API | 固定 ComfyUI commit のソース確認と最小 CPU スタブ |
| Turbo / Raw 推論 | 未実施 |
| FP8 / INT8 / LoRA / img2img | 未検証 |
| CUDA backend / VRAM / 時間 | 未測定 |
| AFM manual / augmented | 未実装 |
| auto 採用プロファイル | 未登録 |

ソース基準:
[Krea2](https://github.com/Comfy-Org/ComfyUI/blob/9d80841aa1990305cc7a280c5ea505f317efcc2d/comfy/ldm/krea2/model.py)、
[Attention dispatch](https://github.com/Comfy-Org/ComfyUI/blob/9d80841aa1990305cc7a280c5ea505f317efcc2d/comfy/ldm/modules/attention.py)、
[wrappers](https://github.com/Comfy-Org/ComfyUI/blob/9d80841aa1990305cc7a280c5ea505f317efcc2d/comfy/patcher_extension.py)、
[samplers](https://github.com/Comfy-Org/ComfyUI/blob/9d80841aa1990305cc7a280c5ea505f317efcc2d/comfy/samplers.py)。

例の workflow は後日のサーバー確認用です。実サーバーの `/object_info` で
loader / sampler / AFM ノードの入力を照合し、インストール済みモデル名へ置換して
初めて実行条件が揃います。静的 JSON 確認を推論成功と扱わないでください。
