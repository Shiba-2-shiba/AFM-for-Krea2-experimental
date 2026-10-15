# 入力契約修正と検証準備の現在地（2026-10-15）

対象は `bbd5a4e417ef122c9c2050b7bbd4777b68b44d4e` からの今回の修正差分。
**修正後のCPU回帰は470成功・GPU 1件除外。GPU推論・実機性能・画質の再検証は未実施。**
過去の22生成条件・人物26画像は修正前の証拠であり、この差分のGPU合格には使用しない。

## 修正した契約

- manual/augmentedの共通入力は同一QKV shape/device/dtypeを要求し、scaleは正の有限なPython数値に限定。bool、文字列、Tensorの暗黙変換はbackend入口でも拒否する。
- `preserve_text_mass` と `auto` は、joint向けprofileの有無にかかわらずmanualへ固定する。
- 例外文言、live schemaを模したrunnerテスト、逐次head処理のpeak workspace期待、LF/CRLF checkoutの固定fixture検査を整合させた。
- spectralの製品計算を変更せず、独立FP64 oracleとの誤差をFP32精度から限定し、checkerboard期待値をsoftmax濃度の解析式で再導出した。一括した許容差緩和ではない。
- legacy計測は実経路を証明できなければunattributedとして表示し、manual/augmentedの資格証拠へ転用しない。

## 次の実験に向けた追加

- preserveだけの対比較に無関係なjoint条件を要求しない。prompt/seed、モデル、sampling、shape、入力画像、設定、実backend/精度/scale等が比較可能か検査し、欠損をunverifiedとする。baseline反復のnoise推定にも同じ来歴の制約を置く。
- original/offやspectral診断offの予定集合は空・unverifiedとし、画像生成をspectral coverage合格へ読み替えない。
- `schedule=frequency_neutral` を任意追加。alphaLF=alphaHF=1+strength/2、共通のDC/LSE処理を使う。総編集量やtemperatureが同じことを意味しない。既定値curve・legacy workflow・空のqualified profileは維持した。
- 主Pilotは512×512、4 prompt×4 seed×5条件の80画像。preserve/manual zeroと強度1/1.5/2、originalを比較する。新しいprompt/seedを事前固定した探索であり、過去の人物比較の完全再現ではない。
- 別matrixのneutral対照16画像、盲検採点テンプレート、[Codex実機検証計画](hardware-validation-plan-2026-10-15.md)を追加。768以上の画質はこの低解像度Pilotでは判断しない。

## CPU検証の環境と再現コマンド

Linux x86_64、Python 3.12.14、PyTorch 2.10.0+cpu、pytest 9.1.1、NumPy 2.5.4。
`KREA2_COMFY_SOURCE_ROOT` はComfyUI `15ef24d1c0333a3eba56c5cd153d8db65363ff8f` のソース契約用treeへ設定した。
これはASTで上流契約を調べるCPU検査であり、同revisionのサーバーを起動した証拠ではない。

```bash
export KREA2_COMFY_SOURCE_ROOT=/path/to/ComfyUI-source-at-15ef24d
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
python -m pytest tests -q
python -m compileall -q afm nodes.py __init__.py scripts
git diff --check
python scripts/validate_workflow.py examples/krea2_afm_edit_api.json
python scripts/run_afm_matrix.py --comfy-url http://127.0.0.1:8188 \
  --matrix examples/validation_matrix_afm_pilot.json --out-dir outputs/pilot-dryrun
python scripts/run_afm_matrix.py --comfy-url http://127.0.0.1:8188 \
  --matrix examples/validation_matrix_afm_pilot_neutral.json --out-dir outputs/neutral-dryrun
```

pytestは470成功・GPU 1件除外。compileall、diff検査、graph検査も成功。
送信なしのdry-runで80件と16件を準備し、全件not_submitted、runtime未確認を維持する。
別担当によるcode reviewとCPU全suite再実行も行った。これはGPU独立検証の代替ではない。
実機実行、モデル取得、既定値変更、auto/profile採用は今回行っていない。
