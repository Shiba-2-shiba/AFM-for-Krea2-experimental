# Krea2 AFM リファクタリング・追加検証計画書

> **For agentic workers:** 実装を別途開始する際は、superpowers:executing-plans または superpowers:subagent-driven-development を用い、以下のタスク単位で実装・検証する。製品コードwriterとreviewerを分離する。本書作成はコード変更・実モデル実験の開始を意味しない。

**Goal:** 現行AFM-curveを維持しつつ、テキスト総注意量を保存するKrea2向け比較方式と、論文との比較・実機互換性・性能を検証できる基盤を整える。

**Architecture:** 既存FFT、manual/augmented kernel、clone/wrapper、atomic fallbackを維持する。共用数値処理だけを整理し、新しい正規化方式はmanualに追加する。診断・集計・profile判定へ方式の識別を通し、数値実装の正しさと実験的な画像効果を別々に判定する。

**Tech Stack:** Python >=3.10、既存PyTorch、pytest、ComfyUI V3 node API。実行時依存の追加なし。

**Spec:** 本書§2–§7を設計仕様とする。旧計画 `Krea2_AFM_Codex_Implementation_Plan_2026-09-30.md` 改訂2を背景資料とし、正規化の選択・診断・検証については本書を優先する。

**作成日:** 2026-10-15 / **状態:** 実装前の計画 / **対象:** AFM-for-Krea2-experimental

## Global Constraints

- 既定動作はjointのまま。新入力がない既存API/UI workflowの計算方式を変えない。
- 追加依存、モデルforwardのコピー、グローバルmonkey patch、汎用attention frameworkを導入しない。
- 画像Query×テキストKeyだけを編集し、全Keyのjoint softmaxを維持する。text-only出力差を無重みで加える近似に置き換えない。
- off/zero originalのno-hook、CFG行保持、txtfusion除外、containerの1回消費、例外時の原子的fallbackを維持する。
- mask、reference連結、複数フレーム、複数GPU、torch.compileの未対応を明示する。
- CPU実装完了とGPU検証完了を分ける。GPU未実施でも実装は完成可能だが、GPU互換性・速度・VRAM・画質は未検証のまま記録する。
- 論文の事実、コードの観測、新しい仮説、実測結果を区別する。

## Review Focus

通常条件の数値一致だけでは見落とす境界。各項目を対応タスクで検証する。

1. 古いworkflow、schema 4ログ、古いprofileで、計算方式・指標の意味をすり替えない（T2/T3/T5）。
2. strength=0/負値、単一text token、ほぼ0/1のtext mass、大きなlogitでも保存性・有限性を満たす（T2）。
3. CFG混在、長方形/奇数latent、端数chunk、途中OOM/NaNで未選択行・baselineを壊さない（T0/T2/T4）。
4. cacheの前後装着、未知sigma、モデル評価省略を正常な全step編集と誤認しない（T4/T6）。
5. 新方式へ旧augmented/profileを流用しない。測定ノイズ、head偏り、画像保存成功だけで合格にしない（T2/T3/T5/T8）。

---

## 1. 出発点と証拠

### 1.1 スナップショット

- 対象HEAD: `944e5074a7a4fd44c248bb1f11b570460c487015`。作成開始時のtracked working treeはclean。
- ローカルComfyUI HEAD: `ba994879b88d0ebd9950b1e5c1e923d928f216ba`。
- 実装の参照基準ComfyUI: `9d80841aa1990305cc7a280c5ea505f317efcc2d`。両revisionを区別する。
- 同スレッドの直前調査で、現ローカルComfyUIをKREA2_COMFY_SOURCE_ROOTに指定したCPU suiteは **241 passed, 1 deselected / 6.68秒**。GPU生成の証拠ではない。
- 本書作成中は製品コードを変更しない。既存記録は `docs/implementation-status.md:133`、`docs/validation.md:18`。

### 1.2 現実装の評価

| 項目 | 判定・確信度 | 根拠 |
|---|---|---|
| Tokenごとの空間FFT、curve、DC保持 | AFM-curveの主要数式に対応 / 高 | `afm/spectral.py:9,42`; 論文式(14)–(16) |
| manual joint attention | 全Key正規化は設計どおり / 高 | `afm/joint_attention.py:37`; 旧計画§3.1 |
| augmented Q/K/V | 固定K・線形空間filterの条件下で代数的に妥当 / 高 | `afm/augmented_attention.py:21,80`; `tests/test_augmented_attention.py:29` |
| Krea2接続 | grid、txtfusion除外、GQA展開後、containerをCPU/ソースで確認 / 高、実機は未確認 | `afm/runtime.py:74,88,242`; `afm/backend.py:30`; `tests/test_source_contract.py:44` |
| 現行HF比 | 平均除去したtext集中度のHF比。論文記載式(7)とは別定義 / 高 | `afm/observation.py:20` |
| cache互換性 | モデル省略cacheとの併用は未検証 / 高。画像への影響は未知 | `afm/capabilities.py:85`; host `comfy_extras/nodes_easycache.py:384,517` |
| 画質・速度・VRAM | 未検証 | README:7; `afm/qualified_profiles.json` は空 |

現方式は画像Keyのlogitを保持しても、共通分母が変わるため画像Keyの確率を変える。これは旧計画§3.1で明示された設計であり、実装ミスと扱わない。今回の目的に対する比較方式として総注意量保存を追加する。

### 1.3 他実装から得た根拠

| 資料 | 確認したこと | 本計画での扱い |
|---|---|---|
| AFM論文v1 [E1] | text-only cross-attentionにlogit FFT。主設定SD1.5/1.4、DDIM 50step、CFG7.5、K=8、cutoff=.25、strength=.2 | 数式・指標の基準。Krea2の画像効果は保証しない |
| Anima AFM [E2] | image query/text keyのcross-attentionでFFT後にtext軸softmax | 論文に近い正規化。画像とのmass補正が不要な構造 |
| Krea2T Enhancer [E3] | 指定text keyへlog(w)を加え全Key softmax。追加Q/K次元で表現 | 現joint方式の直接の先例。総注意量保存の先例ではない |
| TIDE論文・ComfyUI port [E4/E5] | text/image配分を扱い、portはtext biasをSDPA maskに加算 | 配分とtext内部を分ける参考。AFM/Krea2の実証ではない |

TIDE portのbias適用位置は論文式(19)の文字どおりのscale順序と同一ではない。公式実装への遡及も未確認。係数の無検証転用、ソースの丸写しは行わない。

## 2. 目標・非目標・合格の分離

**必須:** 現jointの回帰固定、共用数値処理の小さな整理、manualの総注意量保存mode、明示的な診断定義、旧workflow/log互換、比較実験の同定と集計。

**初回対象外:** entropy gate、TIDE型text anchor、image→image FFT、reference/video対応、既定mode変更、全runtime再設計、baseline計算の削除。新方式augmented高速化はT8の後続候補。

合格は3種類に分ける。

- **数値合格:** 数式・保存性・fallback契約に合う。
- **接続合格:** 指定host/model/backend/shapeで予定した編集が実際に適用された。
- **効果評価完了:** 事前に決めた比較の差・不確実性を報告した。期待方向が出ない結果も評価完了であり、自動的にコード不具合としない。

「論文に近い」は、まずtext内部のtoken competitionという操作対象が近いという意味。終盤HF比の低下や最終画像の改善を数式から保証しない。

## 3. 新方式の数理仕様

### 3.1 定義と補正式

各batch/head/image-queryについて、scale適用済みtext logitをL_T、image logitをL_I、既存FFT operatorをF_uとする。

- baseline: softmax(concat(L_T, L_I))
- joint: softmax(concat(F_u(L_T), L_I))
- preserve_text_mass:

~~~text
z0 = logsumexp(L_T, dim=text_keys, keepdim=True)
E  = F_u(L_T)
c  = z0 - logsumexp(E, dim=text_keys, keepdim=True)
P* = softmax(concat(E + c, L_I), dim=all_keys)
~~~

cは画像Query/headごとのscalarでtext列すべてに同じ値を加える。scale適用後のlogitで計算する。

### 3.2 不変条件と限界

sum(exp(E+c))=sum(exp(L_T))により、**同一QKVの同じattention call内**では:

1. text総確率m_T=Z_T/(Z_T+Z_I)を保存。
2. 各image-key確率を保存。
3. text内条件付き確率はsoftmax(E)。
4. 全Key確率和は1。
5. 出力は `O_base + m_T * ((softmax(E)-softmax(L_T)) @ V_T)` と等価。

5は検証用恒等式。初版はfull-key manual softmaxを維持し、m_Tを省いたtext-only差分加算にしない。

- 後続block・次step・最終negative予測まで固定する保証はない。
- preserve_dcはFFT結果EのDC保持。後段cは空間的に変化し得るため、最終補正logitのDC保持とは呼ばない。text内softmaxにはcが消える。
- F(QKᵀ)=F(Q)Kᵀは線形Fの性質。LSE補正は入力依存なので、現行2d拡張だけでは新方式を自動的に実装できない。
- mask、text tokenゼロ、非有限logitは初版対象外。-infをFFTに渡したり、-inf - -infを計算しない。

### 3.3 合成確認の回帰化

調査時FP32例: seed=79、B=1/H=2/I=96/T=16/d=32、grid=(8,12)、strength=.2、u=.75、cutoff=.25、soft mask幅=.05、DC保持。
jointのtext mass最大変化=.1046334、保存方式誤差=8.94e-8、image確率誤差=1.49e-8、出力恒等式誤差=2.98e-7。

これは1つの合成例であり、GPU/modelの上限誤差・画像品質を推定する値ではない。回帰テストはこの値への完全一致でなく、不変条件と独立oracleへの許容差で判定する。

## 4. API・計算経路・構造整理

### 4.1 設定と互換性

- `AFMConfig.normalization_mode: str = "joint"`。許容値joint / preserve_text_mass。
- normalization_modeとdiagnostic_head_indicesはV3 schema上でoptional=Trueとし、advanced widgetを**既存widgetの末尾**、execute引数も末尾に追加する。execute側の既定値もjoint/空文字に固定する。ローカルComfyUIのcomfy_api/latest/_io.py:164,328,350にoptional引数がある。
- 旧HEADのAPI/UI JSONを改変せずtests/fixtures/legacy_944e507/へ保存し、hashを固定する。旧APIを現object_infoで照合し、両新入力がなくてもexecuteがjoint/従来head選択になること、旧UI再保存で旧widget値を保持することを確認する。
- tests/test_workflows.py:26の全field完全一致は、required集合を全て含み、optionalの省略だけを許す検査に変更する。既存required欠落・未知入力を引き続き拒否し、validatorの厳密性を下げない。新workflowは全設定明示を別テストで要求する。
- diagnostic_top_kの既定4は維持。論文比較用の新workflowは8を明示する。
- T3で `diagnostic_head_indices: str = ""` を追加。空欄は従来の先頭head観測。指定時は実head番号を使い、chunk上限と総観測head数を分離する。

### 4.2 active editの経路選択

| normalization | implementation | 動作 |
|---|---|---|
| joint | manual/augmented/auto | 現行規則 |
| preserve_text_mass | manual | 新manual |
| preserve_text_mass | auto | manual、理由normalization_requires_manualを記録 |
| preserve_text_mass | augmented | 明示unsupported。未補正augmentedを実行しない |

zero controlは既存どおりzero_strength_modeが優先。new mode + zero manualは対応、zero augmentedは未対応。off/zero originalは既存no-hookのまま。

compare_kernels=trueは当面jointのmanual/augmented比較専用。preserve_text_massで有効な診断経路に入る組合せは明示エラーとする。新方式はmass invariant診断とCPU oracleで検証し、異なる方式同士をkernel誤差と呼ばない。

### 4.3 最小限の責務整理

新規 `afm/attention_math.py` に共用入力検証・有限性検査・option拒否・LSE補正を置く。

~~~python
validate_joint_inputs(q, k, v, layout, chunks, scale, implementation) -> tuple[int, int, int, int]
reject_attention_options(mask, is_causal: bool) -> None
require_finite(value: torch.Tensor, label: str) -> None
preserve_text_partition(edited_logits: torch.Tensor,
                        original_log_partition: torch.Tensor) -> torch.Tensor
~~~

最後の関数はFP32入力を破壊せず、補正済みFP32 Tensorを返す。original_log_partitionは[B,H,I,1]。元logitが生きている間にcallerがz0を計算する。

augmentedがjoint_attentionのprivate validationをimportする依存を解消する。FFTはspectral.py、runtime状態はruntime.pyに残す。汎用strategyやデータ型だけの新階層は作らない。

### 4.4 メモリと性能

- z0、c、補正出力、診断で延長保持するTensorをmemory.pyのpeak生存見積もりへ加える。
- ChunkPlan/APIは維持し、configでscratchを見積もる。FFTは全画像grid、その後query分割。
- 予算不足ではchunk縮小→memory_guard。partial editを返さない。
- GPU同期を伴うfinite検査、小さいSDPA反復、baseline＋編集の二重計算を測定する。測定前に安全検査やbaselineを削除しない。

## 5. 診断・ログ・profile

### 5.1 HF指標の明示

同じtext内softmaxからtop-K平均集中度Sを作り、before/afterの各指標を別名で記録する。

| 名前 | 定義 | 用途 |
|---|---|---|
| rho_text_raw | SのFFT power、DCを含む直接cutoff | 論文式(7)＋直接cutoff版の式(10)に対応 |
| rho_text_centered | S-mean(S)のFFT power | 現行指標との連続性 |
| rho_text_radial_binned | radial bin集約後にHF binを合計 | 論文式(9)(10)との比較 |

計算はbranch/headごとに行う。U=abs(FFT2(S))^2、P=U/sum(U)とし、rho_text_raw=sum(P[r>cutoff])。centeredはS_c=S-mean(S)、U_c=abs(FFT2(S_c))^2、P_c=U_c/sum(U_c)、rho_text_centered=sum(P_c[r>cutoff])。binnedはrawの正規化済みPを各binへ和算してE_bを作り、rho_text_radial_binned=sum(E_b[bin_center>=cutoff])。未正規化powerをhead間でpoolしてから割らない。

論文の実際のbin数・境界との厳密一致は未確認。初回はB=32、等幅[0,1]、代表半径=bin中央、上端1=最終binと明記し、16/64 binで感度を確認する。raw直接cutoffはr>cutoff、binnedはbin中央>=cutoffと定義する。

centeredのpowerがraw powerに対して1e-12以下なら数値的に未定義とし、値null・metric別defined=falseを記録する（raw power=0も未定義）。T<=Kはconstant_topkを明示し、raw/DCの値は残すがtoken competitionの統計から除外する。centeredの未定義をraw/binnedへ一律に伝播させない。

- 新JSONLはschema_version=5。metric version、K、grid、cutoff、bin数、head番号、normalization mode、実edit有無を記録。
- v5で旧rho_text_conditional名を再利用しない。v4 readerはcentered legacyとして表示し、raw/binned欠損値を捏造しない。
- 8x8/8tokens/topK4、token0=1+.01*checkerboardの合成例（raw≈3.83e-6、centered≈1）を回帰化。

### 5.2 保存性とhead sampling

text mass平均before/afterに加え、同一QKVのtext_mass_max_abs_delta、image_probability_max_abs_deltaをbounded診断で記録。平均差だけで保存を判定しない。afterは**c補正後**を使う。

diagnostic_head_indicesはcomma-separated実head index。重複/負数/範囲外を拒否し、指定headをhead_chunk_sizeごとに処理する。spectralイベントはbranch/headごとのscalarとして元head番号を記録し、異なるchunkの集約値だけを出さない。診断scope/step間隔/query chunk/workspace guardを維持する。診断on/offで生成出力が変わらないことを確認する。

同一callのafter-beforeはdirect_delta_rho、別runのtarget_after-baseline_afterはtrajectory_delta_rhoとして分ける。編集runのbeforeは既に変更されたtrajectory上の値であり、別runのbaselineではない。mass保存性は同一callで判定し、別run間のmass不変を要求しない。

### 5.3 方式・資格・証拠の混在防止

backend._runtime_signatureのfilter_signatureにnormalization_mode、normalization_version=1を追加。FFT自体のradial_v1は維持。既存selectionのfilter_signature全体一致でmodeを分離する。

- 方式記載のないlegacy reportは表示対象だけ。新profile資格に使用しない。
- mode、model/checkpoint、host/PyTorch/CUDA/GPU、activation/SDPA dtype、shape/chunk/filterの異なる証拠を結合しない。
- qualified_profiles.jsonは空を維持。manual合格でaugmented profileを登録しない。
- benchmarkにはauto指定でなく実際の選択経路を記録。経路混在runを単一kernel比較に使わない。

## 6. runtime・cache・fallback

| 分岐 | 分類 | 方針 |
|---|---|---|
| OOM/NaN/unsupported時fallback_original | 根拠のあるfail-safe | 同じcallのbaseline、partial破棄、理由/回数を維持 |
| 未資格auto→manual | 根拠のある選択規則 | 実行前選択を維持。失敗後の別kernel再試行と区別 |
| GPU未計測/ログ欠落 | 検証状態 | null/unverified。成功で埋めない |
| cacheによるモデル省略 | 未定義の併用 | 正常編集として数えず、既知cacheは事前検出 |
| 未知sigma/CFG metadata | 契約外入力 | 最近傍step/positiveへ推測補完せずraise/fallback |

現hostではEasyCacheがeasycacheをOUTER_SAMPLE/CALC_COND_BATCH/DIFFUSION_MODEL、LazyCacheがlazycacheをOUTER_SAMPLE/PREDICT_NOISEに登録する。DIFFUSION_MODELだけで検索しない。

active edit/診断時は既知cacheをpatch時・run入口で検出しunsupported_model_skip_cacheで停止。AFM前後の装着順を検証。off/zero originalは妨げない。cache検出をattention-call fallbackに変えて実験を続けない。

未知wrapperをすべて拒否するallowlistは作らない。pass-throughは許可し、未知の省略は期待step/blockの観測coverage不足として実験をunverifiedにする。全種類のcache検出を保証しない。

## 7. 追加検証と判定値

### 7.1 数値

| ID | 検査 | 完了条件 |
|---|---|---|
| N1 | joint/zero original/off | 既存241件相当の回帰維持。backend回数・未選択行一致 |
| N2 | mass保存、各image確率、text条件付き分布 | 独立FP64 oracle内部の恒等式/解析解のmax abs<=1e-10。production FP32対FP64 oracleおよびFP32内保存誤差<=1e-5 |
| N3 | 出力恒等式、chunk不変 | FP32 assert_close(atol=1e-5, rtol=1e-5) |
| N4 | schedule、strength、DC/soft mask | strength=[-1,-.2,0,.05,.1,.2,2]、u=[0,.5,1]。FFT段DCと最終補正を区別 |
| N5 | 極端分布/退化 | T=1/T<K、massほぼ0/1、logit±80で有限。条件付き分布は不安定なP_T/m_Tでなくsoftmax(L_T)から計算 |
| N6 | 独立oracle | production FFT/補正helperを呼ばない小grid明示DFTまたは独立FP64計算＋解析的単一周波数 |
| N7 | 低精度出力 | FP32作業scoreの保存と出力castを分離。既存relL2/scaledMax閾値FP32=(1e-4,1e-4)、FP16=(.005,.01)、BF16=(.02,.05)を暫定継承 |
| N8 | 未対応/中途失敗 | 新方式で旧augmentedを無言実行しない。失敗時baseline bitwise一致 |

production spectral.py:56はFP32へ変換するため、production出力へFP64の1e-10を要求しない。FP16/BF16は作業値をN2、cast後出力をN7で別々に判定する。

effect_rel_l2<=.25、effect norm > max(1e-6,3×zero path error)も維持。noise floor以下はunresolvedであり、効果ゼロ/改善とは断定しない（afm/metrics.py:21）。低精度閾値は実モデルの保証ではない。

### 7.2 接続

- 基準commitと現host commitの両方でAST source-contract。host更新時にimport/schema/dispatcherを再確認。
- 512² smoke、768²、768×512、奇数latent/padding、4D/単一frame5D。reference/複数frame/maskは拒否確認。
- Turbo通常設定、Raw/Base CFG>1、positive/negative/both、batch>1、CFG別forward。
- 初回Euler/simple。追加samplerは実評価sigmaを検査し、未知中間sigmaを勝手に補間しない。img2img/denoise<1、start/end境界、重複/非単調schedule。
- cache前後、pass-through wrapper、連続run/例外後run、clone状態分離。
- backend/FP16/BF16/量子化は個別構成。重み量子化とattention dtypeを区別。
- edited_calls>0だけで合格にしない。予定step×block×CFG行とcoverageを照合し、全fallback・欠落を不合格またはunverifiedとする。

#### coverageの契約

最初の対応protocolは `euler_simple_single_eval_per_branch_v1`（batch=1、単一conditioning領域、Euler/simple）。期待semantic key E=(step_index, block_index, branch_kind, head_index)をschedule・diagnostic scope・選択CFGから実行前に作る。各keyにexpected_edit（scope/start/end/branchで決定）、normalization_mode、実装経路を持たせる。

観測physical key A=(step_index, model_eval_index, block_index, batch_row, head_index)も保存する。attentionイベントのcond_or_uncond、batch数とrow対応からbranch_kind（0=positive、1=negative）へ写し、batch=1では各Eに1件を要求する。CFGが同一forward/別forwardのどちらでもpositiveのmodel_eval_indexを0へ固定しない。一般batch>1ではsample_indexをsemantic keyに追加する対応testを用意し、主画像実験とは分ける。

| 状態 | 判定 |
|---|---|
| Eの全要素に一意の観測があり、expected_edit/mode/実経路一致、finiteな必須metric、fallbackなし | coverage=pass |
| 既知protocolで欠損、physical/semantic重複、mode/経路/edit不一致、fallbackあり | coverage=fail、該当runを接続合格・primary集計へ使わない |
| eventはあるが必須metricがundefined | coverage=unverified_metric。原因を残し、そのmetricの統計へ使わない |
| cond_or_uncond/row対応不足、model_eval_index欠損または対応不能、protocol未対応 | coverage=unverified。推測でmappingしない |

診断scope外/編集区間外を誤って欠損としない。区間外でも診断を予定したkeyはexpected_edit=falseを要求する。余分な診断scopeは明示して別集計し、未知eventでEの穴を埋めない。mode/finite/coverageの確認はspectralイベント単独でなく、対応attentionイベントのreason・実経路とrun_summaryを突き合わせる。

### 7.3 attention・画像の実験順序

以下はGPU実験を別途開始する段階の手順。本書作成ではdownload・推論しない。

1. **Smoke:** Turbo/512²/同prompt・seed。baseline×2、discover、observe、zero original、joint manual zero、preserve manual zero、joint .05、preserve .05。画像とログ/coverageを確認。
2. **基準の発見:** 4prompt×2seed。全body block/1headを観測後、候補blockで等間隔4headを確認。Krea2自身のcoarse-to-fineを検査し、先頭4blockをU-Net encoder相当と決めつけない。
3. **Pilot:** 8prompt×2seed、baseline＋2方式×strength(.05,.1,.2)=7条件の112生成に、各pairのbaseline再実行16生成を加え、**計128生成/モデル・解像度**。repeat_index=0をpaired比較用、1を同prompt/seedのbaseline変動推定用とし、結果を見て基準画像を選ばない。人物、動物、文字、texture、複数物体・位置関係を含むpromptとhashを事前固定。
4. **確認実験:** Pilotと重複しない20prompt×4seed=80pair。Pilotで固定したstrength/blockによるbaseline/joint/preserveの3条件、計240生成/モデル・解像度。Turbo/Raw、正方形/長方形は別stratum。全組合せを最初から実行しない。
5. **頑健性:** cutoff=.20/.25/.30、bin16/32/64、K=4/8、soft/hard、DC有無を1因子ずつ。primary以外は探索結果と表示。

primary attention指標はrho_text_raw_afterのtrajectory_delta_rho_late（u>=.8）。K=8/cutoff=.25、Pilotで固定したblock/headで、head→block→stepの順に等重み平均してprompt/seedペア差を計算する。centered/binned/direct_deltaはsecondary。8step時はstep index 6/7の2点となることも記録する。

画像/trajectoryの主実験はbatch=1・positive_onlyを基本とし、CFG/モデル差は別stratumにする。attention測定用baselineはobserve＋spectral診断を使い、Smokeでtrue baselineとの出力同等性を確認する。実験前にexpected step/block/head/branch集合を予定scheduleと設定から作り、実測ログだけから期待集合を逆算しない。未知samplerの期待集合が定義できない場合はcoverage未検証とする。

contrastはpreserve−baselineをprimary、joint−baselineとpreserve−jointを事前指定secondaryとして別々に報告し、モデル/解像度/CFG等のstratumを混ぜない。各prompt内でseedのpaired差を先に等重み平均し、そのprompt平均の平均をestimandとする。

95% CIはprompt平均を復元抽出するcluster bootstrap（10000回、random.Random(bootstrap_seed)、既定20261015）。各回のprompt平均の平均を並べ、percentile法の2.5/97.5百分位を線形補間（index=p*(B-1)）で得る。CIを複数出しただけで多重比較を考慮した有意性とは主張しない。符号率、text mass、0効果/欠損率を併記し、事後除外で有意差を作らない。

各metricは予定されたhead/block/stepを等重み集約する前にdefinedを確認する。1点でも必須metricがnull/欠落ならそのrun/metricは集約不可とし、平均の分母を黙って減らさない。主比較は全3条件が揃う共通complete pairで計算し、期待pair数・実数・除外理由を報告する。centeredのnullはraw primaryを失格にしない。確認実験のprimary pair欠損があればCIは探索的として示し、既定採用を決めない。

画像評価はpaired LPIPS、CLIP、視覚的破綻・構図/文字/複数物体の観察を分ける。LPIPS増加は変化量であって品質向上ではない。評価器のversion/preprocessを固定する。評価ライブラリがなければ追加導入せず未測定とし、外部評価環境の結果をhash付きで取り込む。

終盤rho低下だけで既定採用しない。実用上の非劣性marginはPilotのbaseline反復変動を基に数値固定し、確認実験前にmanifestへ記録する。未設定なら既定採用を判定しない。逆方向の結果では定義・scope・dtype・baseline傾向を確認し、係数変更は別実験として扱う。

### 7.4 性能

診断off、同一model/shape/mode/seed、各方式1warm-up後5回以上、交互順。server内CUDA同期でwall/edit time、peak allocated/reservedを計測しcold/warmを分離。

初版preserve manualはbaseline/joint manualとの差を報告するだけで、速度改善を合格要件にしない。高速化profileは同一modeのmanual対augmentedで、既存5%以上改善・他指標非悪化・変動幅条件と数値合格を要求する。

## 8. 実装タスク

各タスクは回帰/失敗テスト→最小変更→targeted検証→独立レビュー。commitする場合は単独で戻せる単位にする。

### T0. 現行動作を固定

**Files:** tests/test_joint_attention.py、test_augmented_attention.py、test_branch_preservation.py、test_backend.py、test_runtime.py、test_source_contract.py、docs/validation.md。
**Consumes:** 現HEAD。**Produces:** joint/zero/fallback/layout/containerの回帰suiteと環境記録。

- [ ] 既存suiteを再実行し、baseline/host commit・環境を記録。
- [ ] 未カバーの旧入力省略、端数chunk、CFG行、例外後runを既存fixtureに追加。
- [ ] 失敗が既存不具合か新仕様未実装かを区別。
- [ ] `python -m pytest tests -q`。既存テストを削除して数だけ合わせない。source-contract skipは別記。

### T1. 共用数値処理を整理

**Files:** Create afm/attention_math.py; Modify afm/joint_attention.py、afm/augmented_attention.py; Test tests/test_joint_attention.py、test_augmented_attention.py。
**Produces:** §4.3のvalidation/finite/option関数（補正関数はT2）。

- [ ] 例外reason、shape、finite、mask拒否の両kernel契約を固定。
- [ ] private関数依存を共有moduleへ置換。演算順序・dtype・scale・chunkを変えない。
- [ ] `python -m pytest tests/test_joint_attention.py tests/test_augmented_attention.py tests/test_backend.py -q`。T0の出力・呼出契約を維持。

### T2. manual総注意量保存mode

**Files:** Modify afm/config.py、nodes.py、afm/attention_math.py、afm/joint_attention.py、afm/selection.py、afm/backend.py、afm/memory.py、tests/reference.py; Create tests/test_mass_preservation.py、tests/fixtures/legacy_944e507/; Extend tests/test_node.py、test_workflows.py、test_selection.py、test_memory.py、test_branch_preservation.py。
**Consumes:** T1。**Produces:** normalization_mode、preserve_text_partition、§4.2の事前拒否を含む選択規則/signature。head選択入力の追加自体はT3で同じoptional/末尾手順を使う。

- [ ] N2–N8の失敗テストを先に追加。production helperと独立したoracle、分解恒等式で検算。
- [ ] 旧HEAD固定fixtureでoptional省略/object_info/execute/UI widget保持をテスト。既存required欠落と未知fieldの拒否も固定する。
- [ ] test_preserve_auto_ignores_qualified_joint_profile、test_preserve_zero_augmented_rejected_before_kernel、test_preserve_compare_kernels_rejected_before_kernelを追加。mock kernelの呼出回数0と明示reasonをassertする。
- [ ] FFT前にz0、FFT後にcを計算。全Key softmax、baseline行置換を維持。
- [ ] scratch/peak見積もり、境界予算、chunk縮小、memory_guardをテスト。
- [ ] 異なるmode/旧signatureのprofileをmatchさせない。
- [ ] `python -m pytest tests/test_mass_preservation.py tests/test_selection.py tests/test_memory.py tests/test_node.py tests/test_workflows.py tests/test_branch_preservation.py -q`。N1–N8と既存joint suite合格。

### T3. 診断・logの明確化

**Files:** Modify afm/observation.py、afm/diagnostics.py、afm/config.py、nodes.py、afm/backend.py、afm/memory.py、scripts/summarize_krea2_afm.py; Extend tests/test_observation.py、test_reports.py、test_memory.py、test_node.py。
**Consumes:** T2。**Produces:** schema5、§5のmetric/保存性/head選択。

- [ ] 定数/DC-only/単一LF/HF/checkerboard、K>=T、zero powerを独立期待値で検証。
- [ ] after massが補正後、最大誤差が平均で隠れない、diag on/off出力不変、指定headがchunkで切られないことをテスト。
- [ ] v4 centered legacy、raw欠損null、新旧/mode混在の資格拒否を実装。
- [ ] T2で導入したnew+compare_kernelsの事前拒否を維持し、jointの同一QKV比較を回帰確認。
- [ ] `python -m pytest tests/test_observation.py tests/test_reports.py tests/test_memory.py tests/test_node.py -q`。定義/schema/出力/memoryの全条件合格。

### T4. runtimeの省略・未対応を明示

**Files:** Modify afm/capabilities.py、afm/runtime.py、afm/backend.py; Extend tests/test_capabilities.py、test_runtime.py、test_source_contract.py、test_backend.py。
**Produces:** §6の検出とcoverage証拠。

- [ ] 実keyを使った最小host stubでEasyCache/LazyCache両装着順を再現。pass-throughは許可する対照。
- [ ] 中間sigma、denoise<1、CFG別forward、連続run、上流契約をテスト。
- [ ] 既知cacheをpatch/run入口で拒否。未知省略によるcoverage不足を実験判定へ伝える。
- [ ] unsupported理由の二重計数、profile_not_qualifiedのsummary欠落は、1callの期待値を固定して修正。無関係なlogger再設計をしない。
- [ ] `python -m pytest tests/test_capabilities.py tests/test_runtime.py tests/test_backend.py tests/test_source_contract.py -q`。host2revisionの証拠を分ける。

### T5. 比較実験・集計の再現性

**Files:** Modify scripts/run_afm_matrix.py、scripts/summarize_krea2_afm.py; Create scripts/analyze_afm_experiment.py、examples/validation_matrix_afm_refactor.json; Extend tests/test_matrix_runner.py、test_reports.py; Create tests/test_experiment_analysis.py; Update docs/validation.md、README.md、docs/implementation-status.md。
**Consumes:** schema5。**Produces:** manifest2、paired差/欠損/coverage/CI集計。既存matrix入力v1とsmokeは維持。

manifest追加項目: pair_id、condition_id、repeat_index、phase、prompt_text_sha256、resolved_workflow_sha256、model識別、実経路、metric定義、期待coverage、runtime run_id。既存invocation_id・固有log pathも保持。

pair_idはpositive/negative text、seed、checkpoint/encoder/VAE識別、LoRAと重み、sampler/scheduler/steps/CFG/denoise、resolution、入力画像hash（ある場合）のcanonical JSONから作る。mode/strength/scope/filterはcondition_idに分離する。log先・保存prefix・invocation_id・AFM入力を含む全workflow hashをpair_idへ流用しない。resolved_workflow_sha256は条件ごとの追跡専用。

~~~python
# scripts/analyze_afm_experiment.py
analyze_experiment(manifest: dict, reports: list[dict],
                   *, bootstrap_seed: int = 20261015,
                   bootstrap_samples: int = 10000) -> dict
~~~

- [ ] 同prompt/seedでもmodel/resolution/sampler違いをpairにしない。欠損/duplicate/stale log/partial fallback/metric混在をテスト。
- [ ] raw/centered/binnedを混ぜずpaired差・coverage・undefined件数・prompt-cluster CIを集計。手計算可能なsynthetic reportとseed再現性を検証。
- [ ] benchmarkは同一modeの実経路、実行順/warmup/shape等の証拠で結合。単なる配列連結で資格化しない。
- [ ] 新workflowはmode明示/K=8。旧API/UIのjoint互換を維持。matrixのdiagnostic labelだけでなく実node inputを設定。
- [ ] 新条件dry-runでprompt/manifestのmode、strength、pair、hash、log先一致を確認。
- [ ] `python -m pytest tests/test_matrix_runner.py tests/test_reports.py tests/test_experiment_analysis.py tests/test_workflows.py -q`。dry-runはnot_submitted、GPU値unverified。

### T6. 実ComfyUI接続（GPU環境がある段階）

**Files:** Extend tests/gpu/test_krea2_afm.py; 結果outputs/krea2_afm_refactor/。
**Consumes:** T0–T5合格、選定server/model。**Produces:** §7.2構成別結果。

- [ ] object_info保存、新旧workflow照合、import/node表示/UI保存再ロード。
- [ ] Smoke→rectangle→CFG>1。GPU testを画像保存だけでなくmode/実経路/coverage/finite/保存性のassertへ拡張。
- [ ] 全fallback/ログ欠落を合格にしない。構成別に未実施を記録。
- [ ] 環境がなければ必要条件を明記し、CPU実装完了と実機未検証を分ける。

### T7. 効果の方向・範囲

**Files:** manifest、画像、JSONL、集計report; 結論docs/validation.md。
**Consumes:** T6の該当構成合格。**Produces:** §7.3のdiscovery/pilot/確認結果。

- [ ] 条件/primary metricを先に固定。探索と確認promptを分離。
- [ ] baseline傾向、同一QKVへの直接効果、trajectory間差を別々に評価。
- [ ] 画像評価器version・入力hash・前処理を記録。未測定を0にしない。
- [ ] 既定採用/実験mode維持/仮説不支持を、CI・破綻例・costとともに判断。追加係数は次実験。

### T8. augmented・性能改善（後続、初回完了に不要）

**Files:** afm/augmented_attention.py、afm/memory.py、afm/selection.py、tests/test_augmented_attention.py、test_memory.py、test_reports.py。
**Gate:** 新manualの数値・接続合格、測定で改善対象を特定。

- [ ] LSE補正をchunked additive maskまたは追加定数Q/K次元で表現する案を別途設計。scale補償を含める。
- [ ] Krea2 d=128では現拡張2d=256、追加1次元で257となり得る。kernel対応/整列/padding/V次元/元scale/fallback/VRAMを調べ、Flash使用を推測しない。
- [ ] 同一modeのmanual FP32と同一QKV、zero/nonzero/保存性/数値閾値を検証後に性能測定。
- [ ] 資格にmode/versionと全環境を固定。満たさなければautoはmanual、profile未採用のまま完了可能。

## 9. コマンドと成果物

以下は実装時の例であり、本書作成時の実行済み一覧ではない。

~~~powershell
$env:KREA2_COMFY_SOURCE_ROOT = (Resolve-Path '../ComfyUI').Path
python -m pytest tests -q
python -m compileall -q afm nodes.py __init__.py scripts
git diff --check
python scripts/validate_workflow.py examples/krea2_afm_edit_api.json
python scripts/run_afm_matrix.py --comfy-url http://127.0.0.1:8188 --matrix examples/validation_matrix_afm_refactor.json --out-dir outputs/krea2_afm_refactor/dry-run
~~~

ruff/mypy/pyrightが既存環境にあれば対応scopeで実行。なければ未実施と記録し、追加導入しない。構文検査を型検査合格とは呼ばない。

成果物は、回帰suite・独立oracle・保存性テスト、schema5/v4 reader、modeを分ける資格/実経路記録、新旧workflow、dry-run/paired集計、更新README/validation/implementation-status。実機実験を行った場合だけ、model/backend/commit付きログ・画像・CSV/JSON・CI・失敗例を加える。

## 10. 完了・停止・引き継ぎ

- **初回実装完了:** T0–T5合格、joint互換、数値保存性、明示診断、ログ分離、重大レビュー指摘解消。実機未実施は明記。
- **対応表明:** T6で通したmodel/backend/shapeだけ。
- **効果表明:** T7の範囲だけ。複数seed画像差だけで一般的改善と称さない。
- **自動高速経路:** T8の同一modeに対する数値・性能証拠が揃うprofileだけ。
- 保存性が不成立なら公開前にscale/slice/FP32対FP64 oracleを調査。画像効果が逆なら結果を保存し仮説更新。
- 製品writerとreviewerを分離し、数理/互換/測定をレビューする。テスト数だけで承認しない。

## 11. 出典

ローカルfile:lineは§1のHEADでの位置。実装後はsymbol名も併用して追跡する。

- [E1] [AFM v1](https://arxiv.org/html/2603.28114v1)。ユーザー提供同名PDF全16ページを確認。式(7)–(10)、(13)–(18)、§4.1–4.3。
- [E2] [Anima AFMの確認したmain](https://github.com/Shiba-2-shiba/AFM-for-anima-experimental/blob/main/anima_afm.py)。旧計画参照revision=596abc87f4db983aaceea19b437119367f45eb23。今回webでは固定blobとのbyte一致未確認。
- [E3] [Krea2T Enhancerの確認したmain](https://github.com/capitan01R/ComfyUI-Krea2T-Enhancer/blob/main/attention_weighted_phrase.py)。旧参照revision=e1f60f79e168c62b1cb1cf773dfb3ae4ee2292e2。今回固定blobとのbyte一致未確認。
- [E4] [TIDE v1](https://arxiv.org/html/2603.08928v1)、式(11)–(19)。AFMのmass保存案そのものの出典ではない。
- [E5] [ComfyUI-TIDE patches.py](https://github.com/xmarre/ComfyUI-TIDE/blob/main/tide_core/patches.py)、[math.py](https://github.com/xmarre/ComfyUI-TIDE/blob/main/tide_core/math.py)。独立port、動くmain参照。
- [E6] [PyTorch 2.10 SDPA](https://docs.pytorch.org/docs/2.10/generated/torch.nn.functional.scaled_dot_product_attention.html)。明示scale/additive maskを確認する一次資料。
- 本書のmass保存補正は本スレッドで導出・合成確認した**新しい移植仮説**。他repoの検証済み機能として帰属させない。

## 12. 計画書レビュー記録

- 2026-10-15: 作成者が数理仕様、API/タスク依存、same-callとtrajectory比較、pair IDとworkflow hashの分離を自己点検。
- 独立criticレビュー初回の必須5点（旧workflowのoptional入力、rho定義、精度別基準、baseline反復/CI、coverage契約）を反映。
- 改訂版の独立再レビューは **APPROVE**。必須問題なし。CIの乱数生成を公開bootstrap_seed引数に合わせる軽微指摘も反映。
- 文書検査: 9タスク・42チェック項目。参照パスを既存と新規作成予定に区別して確認し、未解決参照なし。code fence・末尾空白・OMX入口リンクを確認。
- 今回追加したのは本書と.omx/plansの正本参照のみ。製品コード変更、依存導入、モデルdownload、GPU実行は行っていない。

## 13. 実装着手と今回の適用範囲（2026-10-15）

ユーザーの「計画書にそって実装。検証は別に行うため今回は行わない」という指示により、T0–T5の製品コード・テストコード・集計ツールの整備を進めた。検証コマンドとレビューgateは今回は実行しない。実装状態・変更点・未実施の確認手順は [実装引き継ぎ](../../refactoring-implementation-2026-10-15.md) に記録する。上記の検証checkboxと数値/実機の合格条件は未確認のまま残す。

実装時の補足: 計画にあるradial bin数16/32/64の感度比較を設定できるよう、diagnostic_radial_bins（既定32、1..256）もoptional入力の末尾に追加した。
