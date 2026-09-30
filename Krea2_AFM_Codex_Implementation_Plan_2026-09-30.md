# Krea2 AFM カスタムノード Implementation Plan — 改訂2

> **For agentic workers:** REQUIRED SUB-SKILL: 実装時は `superpowers:executing-plans` または `superpowers:subagent-driven-development` を使い、タスク単位で進める。チェックボックスは実施状況の記録に使う。再開時は `handoff.md` と現在の実装状況を先に読む。

**Goal:** `Shiba-2-shiba/AFM-for-anima-experimental` のAFMの考え方と数値処理を利用し、Krea 2の推論時Attentionを周波数編集できる、独立したComfyUIカスタムノードを設計・実装する。

**Architecture:** Krea2の画像Query→テキストKeyのlogitsだけを周波数編集し、画像・テキストの全Keyを含むjoint softmaxを維持する。直接logitsを編集する分割計算を基準実装とし、同じ線形FFT処理を画像Qに移してQ／Kを拡張する等価な計算法を高速化の優先候補にする。ComfyUIのmodel clone、wrapper、`post_input`、Attention overrideを利用し、baselineから選択branchの画像Query出力だけを置換する。

**Tech Stack:** Python、PyTorch、ComfyUIネイティブKrea2実装、ComfyUI V3ノードAPI、CPUで動くpytest、JSONL。モデル学習・追加モデル・独自CUDA拡張は初版では不要。

**Spec:** 本書「3. 設計仕様」。設計と実装タスクを同じファイルにまとめ、Codexへの引き継ぎ時に参照を失わないようにする。

**作成日:** 2026-09-30、日本時間。

**改訂:** 2026-09-30、改訂2。前版を本書で置き換える。`ComfyUI-Krea2T-Enhancer` の全Key正規化とQ／K拡張を参考に、計算方式・backend選択・数値対照・段階的採用を更新した。

**状態:** Task 0–4とTask 5のオフラインツールを実装済み。実モデルの推論、GPU上の速度・VRAM測定、生成品質の確認は未実施。

**実装進捗追記（2026-09-30、再開後）:** manual/augmented編集、メモリ制御、branch保持、atomic fallback、ノード、API/UI例、集計・matrix runnerを実装。runnerの過去ログ混同とdtype証拠混在を修正し、CPU/上流ソース契約テスト241件PASS、GPU1件は除外。今回のユーザー指定により実モデル検証は行わない。GPU中間判定は対応・性能・profile採用の条件として残し、実装とCPU検証は進める。検証根拠・残課題は `docs/implementation-status.md` を参照。

## Global Constraints

- 新規リポジトリ名の案は `ComfyUI-Krea2-AFM`、ノードIDは `Krea2AFMModelPatch`、表示名は `Krea2 AFM Model Patch`。既存Animaノードとは別パッケージにする。
- 初版はネイティブKrea2の静止画T2Iと通常のimg2imgを対象とする。RawとTurboは同じ構造を対象にできるが、個別に動作確認する。
- 通常の4次元latentと、時間軸が1の5次元latentを対象とする。複数フレーム、reference画像を連結する編集経路、regional promptingは初版の対象外。
- `off` と、既定の `edit + strength=0 + zero_strength_mode=original` はAttention overrideを設置しない。
- 編集対象は画像Query→テキストKeyのlogits。softmaxの分母は全Key。テキストだけのsoftmaxでjoint Attentionを置き換えない。
- FFTはFP32で実行する。画像Queryを空間方向に分割してからFFTしない。
- Q／K拡張は、全テキストKey・全Q特徴次元に共通の線形空間フィルタを使う場合に限定する。entropy gateは初版で無効。拡張後にQK norm／RoPEを再実行しない。
- Attentionの元の実効 `scale` を明示して保持する。明示scaleがなければ元のhead dimensionの `d**-0.5` を使い、拡張後の `2*d` から計算し直さない。
- QK norm、RoPE、GQA展開、sigmoid attention gate、出力投影、AdaLN、残差接続はKrea2本体の実装に任せる。
- モデル重み・共有モジュールを直接変更しない。グローバルな関数の差し替え、重みのdeepcopy、全モデルのFP32化をしない。
- 入力modelの設定を変更せず、cloneにパッチを設定する。実行統計・Tensorキャッシュをモデル間／生成間で共有しない。
- 全joint logitsの密な実体化は小規模な参照テストに限定する。実モデル用はhead／Query分割を実装する。
- Q／K拡張経路でもhead／Query分割を維持し、SDPAのmath経路に切り替わっても全画像Query×全Keyを一括確保しない。通常の編集経路では観測のためのlogits再生成もしない。
- `implementation=auto` は未検証環境ではmanualを選ぶ。augmentedは明示選択による検証後、GPU数値比較と性能条件を満たした構成だけautoで選べるようにする。
- 初版の編集は `mask=None`、非causal、dropout=0に限定する。maskやadditive biasをFFTへ含めない。
- エラーは「対象外」「スコープ外」「リソース不足」「実装上の異常」を区別する。開発時は `fail_mode=raise` とする。
- ComfyUI、PyTorch、CUDA、モデルファイル名、Attention backendをログに残す。ComfyUIの最低対応版は実際に確認したcommitと機能検出で定義し、推測した版番号を記載しない。
- ローカルCPUで数値・互換性テストを行い、クラウドGPUで実モデルを検証する。CPUの成功をGPU対応の根拠にしない。
- 実機評価はA4000 16GB程度の環境も想定するが、初版の動作保証ではない。AFMなしで動く量子化・offload構成を出発点にする。

## Review Focus

1. 長方形・奇数latent寸法・単一フレーム5D入力：実際のpatch後gridを使い、テキストやpaddingを画像gridに混ぜない。Task 1、2でテストする。
2. TurboのCFG=1、RawのCFG batch／別forward：`cond_or_uncond` から対象を決め、未選択branchをその呼び出し内で直接変更しない。Task 4でテストする。
3. INT8などのモジュール別Attention backend、Tensor container：observeや対象外の呼び出しで元のdispatchを維持し、入力を二重消費しない。Task 1、4でテストする。
4. `d→2d`、明示scale、headごとの出力復元、SDPA math切り替え：直接logits編集との等価性と分割上限を検証する。途中OOM・NaNでは部分編集を返さない。Task 3、3A、4でテストする。
5. 同じloaderから複数モデルを分岐、連続生成、例外終了、他Attentionパッチとの併用：設定と統計を漏らさず、競合を検出する。Task 1、4でテストする。

---

## 1. 解析対象と確認した事実

解析時点のソースを固定する。実装開始時にクラウドで使用するComfyUIのcommitも調べ、差分を再確認する。

| 対象 | 確認したrevision／ファイル | 設計上の意味 |
|---|---|---|
| Anima AFM | `596abc87f4db983aaceea19b437119367f45eb23` | 移植元の固定点 |
| ComfyUI | `9d80841aa1990305cc7a280c5ea505f317efcc2d` | 今回解析したKrea2／Attention API |
| Krea2T Enhancer | `e1f60f79e168c62b1cb1cf773dfb3ae4ee2292e2`、`attention_weighted_phrase.py` | Query分割・全Key保持・追加Q/K次元・元scale保持の参考 |
| 公式Turbo workflow | `image_krea2_turbo_t2i.json`、blob `6be92fdf6f62b308e9247d4149f84a1691350c44` | FP8モデル、Qwen3-VL、Qwen image VAE、8 steps・CFG=1・Euler/simpleの例 |

### 1.1 移植元の処理

`anima_afm.py` の `edit_logits_fft()` は、`[B,H,N_image,N_text]` のlogitsを `[B,H,N_text,H_grid,W_grid]` に並べ替えて2D FFTを行う。低／高周波のgainを掛け、逆FFTで戻し、softmax前のlogitsを置き換える。

`schedule_alphas()` の `curve` は、denoising進行度を `u`、strengthを `s` として、entropy gateが無効なら次の式になる。

```text
alpha_lf = 1 + s * (1 - u)
alpha_hf = 1 + s * u
```

`preserve_dc=True` は周波数 `(0,0)` の成分を変更しない。`radial_low_high_masks()`、スコープ指定、CFG branch選択、JSONLログ、observe／discoverの区別は再利用に適している。

一方、`AnimaAFMAttentionOverride._prepare_context()` は `query_len == key_len` の呼び出しを `not_cross_attention` として除外する。Krea2のjoint Attentionは基本的にこの長さが等しいため、名前やモデル判定だけを変えてもAFMは適用されない。

### 1.2 Krea2の処理

`comfy/ldm/krea2/model.py` の `SingleStreamDiT._forward()` は、`txtfusion` と `txtmlp` の後で `torch.cat((context, img), dim=1)` を行う。並びはテキストが先、画像が後。

本体のblockループには `block_index`、`total_blocks`、`block_type='single'`、`img_slice=[txtlen, combined.shape[1]]` が渡る。`post_input` はtxtfusion完了後、joint sequenceの構築前に呼ばれる。

`Attention.forward()` はQK norm→RoPE→GQAのKey/Value展開→`optimized_attention_masked()`→sigmoid gate→`wo`の順で処理する。既定の本体は28 blocks、48 Query heads、12 KV heads、patch=2だが、実装では実際の属性とTensor形状を読み、固定値に依存しない。

現在のAttention APIには `AttentionTensorContainer` と、モジュールごとの `ComfyAttention.function` がある。overrideから単に元関数を呼ぶだけでは、モジュール別backendのdispatchを再現できないケースを考慮する。

### 1.3 Krea2T Enhancerから取り入れる内容

参照箇所は `attention_weighted_phrase.py` の `_make_attention_override()`、`_augment_image_attention_qkv()`、`_strip_augmented_output()`。画像／テキストでQuery行を分け、両方とも全Key／Valueを参照する。画像Queryから対象テキストKeyへの `log(w)` 加算をQ／Kの追加次元で表現し、元のscaleを渡す。これは各Query行の全Key正規化を維持する実装例である。[S1]

通常のEnhancerのtxtfusion調整・RMS制御はAFMの移植対象に含めない。参考コードのphrase biasは位置に依存しないが、AFMは画像位置に依存するため、その追加1次元をそのまま流用することはできない。本書では線形性を使って `d→2d` の拡張を新たに導く。参考リポジトリ自体がAFMを実装・検証しているとは記載しない。

取り入れるのは計算原理であり、他ノードへのruntime依存は追加しない。コードを転用する場合は両リポジトリのLICENSEを確認し、必要な著作権・ライセンス表示を保持する。

### 1.4 AFMで何を期待するか

AFMはテキストと画像位置のAttentionパターンを周波数方向に変える、学習不要の介入である。元論文の実験はStable Diffusionを対象とし、意味的な整合性をおおむね保ちながら見た目を編集する技術として報告されている。

Krea2での細部・質感・構図の変化、プロンプト追従性の改善は検証仮説とする。今回の案はjoint Attentionへの拡張であり、元論文をそのまま再現した実装とは表記しない。

## 2. 計算仕様と実行方式

「画像→テキストだけを周波数編集し、画像＋テキスト全Keyでsoftmaxする」を共通仕様とし、以下の3経路を作る。数学的な編集内容と、計算を実行するbackendを混同しない。

| 経路 | 処理 | 役割 |
|---|---|---|
| `dense_reference` | 全joint logitsの対象領域を直接FFT編集 | 小規模CPUテスト専用。ノードから選択不可 |
| `manual` | headごとの画像→テキストlogitsをFFT編集し、Query chunkごとに全Key softmax | 基準実装、実モデルでの比較対象、互換経路 |
| `augmented` | 画像QをFFT処理し、Q/K/Vを拡張して全KeyのSDPAを実行 | 高速化・省メモリ化の優先候補。GPU検証後に条件付き採用 |

テキストKeyだけでsoftmaxして出力差を加える近似、画像→画像logitsまでFFTする拡張、gate倍率の変更は初版に含めない。

`auto` は3.4.4の選択規則を使う。Q/K拡張が数学的に等価であることだけを理由に、あらゆるGPUで高速・省メモリと宣言しない。

## 3. 設計仕様

### 3.1 編集範囲と正規化

テキスト長を `T`、画像token数を `I`、全長を `N=T+I` とする。Query側とKey側の両方をテキスト／画像で分けると、joint logitsは4領域になる。

| Query → Key | テキストKey | 画像Key |
|---|---|---|
| テキストQuery | 直接編集しない | 直接編集しない |
| 画像Query | **FFT編集する** | logitsを直接編集しない |

```text
L = Q K^T * scale
L_it = L[..., T:T+I, 0:T]
L_it_new = AFM_FFT(L_it, grid_h, grid_w, alpha_lf, alpha_hf)

画像Query行では：
P_img = softmax(concat(L_it_new, L_ii), dim=all_keys)
O_img = P_img @ V_all
```

画像Key側のlogitsを変えなくても、全Keyで再正規化するので画像KeyのAttention確率は変わりうる。この変化は方式に含まれる。テキスト側と画像側のsoftmaxを別々に計算して後で足す方法に置き換えない。

通常backendでbaselineのAttention出力を1回計算し、そのcloneの選択branch・画像Query行だけを `O_img` で置き換える。テキストQueryの出力と未選択branchの出力はbaselineから保持する。ただし後続blockや次のdenoising stepには画像の変化が伝わるため、最終的なテキスト表現／negative予測全体が不変になるとは主張しない。

### 3.2 パッチの設置

1. `model.clone()` に設定する。モデル重みの複製は不要。
2. `WrappersMP.OUTER_SAMPLE` でrun単位の `RunState` を作り、実行時に複製された `model_options['transformer_options']` に渡す。`finally` で集計してTensor参照を解放する。
3. `WrappersMP.DIFFUSION_MODEL` で元latentの寸法、`diffusion_model.patch`、referenceの有無を読む。ローカルコピーした `transformer_options` にlayoutの候補を付加してexecutorを呼ぶ。
4. `set_model_post_input_patch()` で、実際の `txt.shape[1]`、画像token数、`img_ids` の空間順を検証し、本体block向けのoverrideをこの時点で設置する。
5. overrideは本体の `block_type='single'`、`block_index`、`img_slice`、QKV形状を検証して処理する。

**txtfusionにはoverrideを設置しない。** txtfusionは同じAttentionクラスを使うが、画像空間に対応するAttentionではない。遅延設置により、txtfusionのbackendと処理を元のまま保つ。

入力のlatent gridと `img_ids` の両方で空間形状を確認する。現行のpatch処理に対応するgridは `ceil(latent_height / patch)` × `ceil(latent_width / patch)`。長方形もこの実寸で処理し、token総数から平方根や近い約数で推測しない。

既存の `post_input`／QKV／Attention出力パッチがある場合、初版は明示的な競合エラーにする。通常のLoRAの重みパッチは別に扱い、併用試験を行う。後続版で対応する場合も、token順・形状を最終的に検証できる組み合わせだけを許可する。

### 3.3 backendとcontainerの保持

overrideは従来の `__call__(original_func, ...)` と、containerを使う現行APIの `container_function(...)` の両経路を扱う。

元backendは、該当blockの `attn.comfy_attention.function` が設定されていればその関数、なければ標準の `optimized_attention_masked` とする。再帰overrideを避け、元のscale・mask・reshapeオプションを維持する。container対応関数がある場合はその入口を保持する。

編集時は `peek()` で必要なQKV参照を保持した上で、baselineへのdispatchでcontainerを1回だけ消費する。baseline計算後のfallbackで同じcontainerを再び `take()` しない。observe／discover／対象外の処理は、元backendへのdispatchを1回だけ行う。

`off`／既定のzero strengthはパッチを設置しないので、containerやbackendの分岐自体に介入しない。

重みのFP8／INT8とAttention計算の量子化は別に記録する。baselineは元backend、manualはFP32 logits／softmax、augmentedは明示的なPyTorch SDPAを使う。編集経路を元の量子化Attentionへ無条件に渡さない。数値経路による差は方式ごとのzero対照で測る。

### 3.4 Q／K拡張と省メモリ計算

#### 3.4.1 線形FFTをQへ移す

head形式は `[B,H,N,d]`。`T` はテキスト長、`I=grid_h*grid_w` は画像長、`N=T+I`。`A` は画像空間だけに作用する線形フィルタで、同じbranch/head内の全テキストKeyに共通とする。周波数gainは固定された設定・denoising進行度から決める。空間周波数ごとのgainが異なること自体は線形性を妨げない。

```text
L_it = scale * Q_img @ K_text.T
A(L_it) = scale * A(Q_img) @ K_text.T
DeltaQ = A(Q_img) - Q_img

Q_img_aug = concat(Q_img, DeltaQ, feature_dim)
K_text_aug = concat(K_text, K_text, feature_dim)
K_img_aug = concat(K_img, zeros_like(K_img), feature_dim)
V_all_aug = concat(V_all, zeros_like(V_all), feature_dim)
```

このとき、画像→テキストの内積は `scale*A(Q_img)@K_text.T`、画像→画像の内積は元の `scale*Q_img@K_img.T` になる。全Keyを残してSDPAを呼ぶため、softmaxの分母も共通仕様と一致する。これは実数演算での等価変形であり、低ランク近似ではない。有限精度ではFFTと行列積の順序・追加次元・backendによる誤差がある。

適用条件を固定する。

- QK norm・RoPE・GQA展開が完了したoverride入口のQを使用する。拡張後にnorm・RoPE・GQA展開を追加しない。
- Q特徴次元を画像grid上の信号として扱い、同じAを各特徴次元に適用する。テキストKeyごとに異なるgain、入力依存の非線形clip／normalize、entropy gateを追加しない。
- 実行時QがFP16/BF16でも、`Q.float()`、FFT、`DeltaQ` の差分計算までFP32で行う。SDPAへ渡す直前に元のactivation dtypeへ変換する。診断用FP32経路も用意する。
- 元の実効scaleを保持する。明示値がなければ `d**-0.5`。標準のd=128なら拡張後は256だが、scaleは `1/sqrt(128)` のまま。[S1, S2]
- `F.scaled_dot_product_attention(..., scale=original_scale, dropout_p=0.0, is_causal=False)` を利用する。すでにhead数が一致するため、追加GQAは有効化しない。必須APIを機能検出する。
- mask非None、causal、未知の追加logit変換は初版では編集対象外。将来maskへ対応する場合も仕様は `A(raw_logits)+mask` とし、maskの `-inf` をFFTへ入れない。
- 出力はhead形式 `[B,H,Qchunk,2d]` で各headの先頭dだけを残す。その後で元の出力形式へ戻す。flatten済みの先頭 `H*d` 列を取る実装は禁止する。

通常の元Attention出力を保持する方針は3.1のまま。参考リポジトリのようにテキストQueryだけ再計算すると、kernelの形状が変わって末尾の数値まで同一とは限らないため、本書ではbaselineのテキスト行をそのまま残す。

#### 3.4.2 2つの実モデル用kernel

選択branchを1つずつ、headを小グループずつ処理する。各kernelの入力は当該branchのQ/K/V、戻り値は画像行 `[B_selected,H,I,d]`、dtypeは元Vとする。

**manual:** 現在のhead群について全画像Query×全テキストKeyのlogitsをFP32で計算し、全画像gridでFFT編集する。その後Queryを行chunkに分け、各chunkの全Key logitsのテキスト列を置換し、FP32 softmaxとValue積を行う。FFTをQuery chunk内だけで行わない。

**augmented:** 現在のhead群について全画像gridのQをFP32でFFTし、DeltaQを作る。全Keyの拡張K/Vはhead群ごとに一度作り、画像Query chunkの拡張QとSDPAを呼ぶ。追加V次元の出力をheadごとに除いて保存する。通常経路で編集用の `I*T` logitsを明示的に作らない。

両経路ともKey軸は全長Nを保持する。Keyを分けた独立softmaxを足し合わせない。

PyTorch SDPAは入力・環境によって内部kernelを選択し、math経路ではFP32中間配列を使う場合がある。[S2] したがってaugmentedもhead／Query chunkを必須にする。fused kernelが未確認でも、分割したmath経路を予算内で利用できる。通常実行でSDPAのグローバル設定を変更しない。`torch_sdpa` の呼び出し成功だけでFlashAttention使用と記録しない。

#### 3.4.3 作業領域と性能

初期値は `head_chunk_size=2`、`query_chunk_size=128`、`max_workspace_mib=512`。予算を超える場合はhead chunkを1まで、Query chunkを32まで縮める。なお超過すれば `memory_guard` とする。選択branchの全batchを一度に複製せず、必要ならsample単位で処理する。

以下は `B=1,H=48,I=4096,T=512,d=128` の単一Tensorのサイズ。実測VRAMではない。

| Tensor | サイズ |
|---|---:|
| 全joint logits、FP32 | 3,888 MiB |
| 全headの画像→テキストlogits、FP32 | 384 MiB |
| 2 headの画像→テキストlogits、FP32 | 16 MiB |
| 2 headの画像Q、FP32 | 4 MiB |
| 2 head・128 Query行の全Key logits、FP32 | 4.5 MiB |
| 2 headの全Key拡張K、FP16/BF16 | 4.5 MiB |
| 2 headの全Key拡張V、FP16/BF16 | 4.5 MiB |

FFTする信号の本数はmanualのT本からaugmentedのd本へ減る。T=512、d=128なら1/4だが、Attentionの内積次元とpadding付きValue次元は2倍になる。短いpromptでT<dの場合もあり、全体の高速化率・VRAM削減率はこの比から推定しない。

見積もりは生存期間を考慮して、保持QKV、baseline出力、出力clone、画像行バッファ、FP32入力、FFT complex配列、復元配列、DeltaQ、dtype変換、拡張Q/K/V、SDPA mathのscore・確率・FP32中間配列、診断配列を含める。`peek()`したQKVを保持すると通常container backendが解放できたTensorも生存するので、これを除外しない。

`max_workspace_mib` はAFMの追加／延長生存する作業領域の目安。cuFFT・SDPA内部workspaceやallocator差があるため総VRAM上限を保証しない。GPU空き容量も確認し、baseline自体が動く構成を前提とする。毎chunkの `empty_cache()`、全モデルFP32化、大きい自動再試行は行わない。

baseline＋画像行再計算の二重計算は残る。初版ではこの構造を変えず、Task 5で効果・時間・VRAMを測定する。

#### 3.4.4 経路選択・fallback・autoへの採用

- `implementation=manual`：基準kernelを固定使用する。
- `implementation=augmented`：Q/K拡張を明示使用する診断・実験用。未対応条件を別方式へ黙って変更せず、理由を返す。
- `implementation=auto`：初期の採用リストは空でmanual。Task 5で合格した実行条件と形状範囲のみaugmentedを選ぶ。判定は生成開始前／呼び出しの事前確認で行い、毎回benchmarkしない。

採用プロファイルはPython/PyTorch/CUDA、GPU名・compute capability、ComfyUI commit、baseline backend、SDPAの有効kernel設定、activation dtype、元head dimension、対象T/I/batch/head/query chunkの検証範囲を記録する。未知の組み合わせ、範囲外形状、空間フィルタの仕様変更ではmanualへ戻し、理由をログに残す。完全一致を満たさないプロファイルを推測で横展開しない。

既知の未対応条件や事前のメモリ見積もり不足はautoの経路選択に使える。一方、実行途中のOOM・NaN・未知例外を握りつぶしてmanualへ連続再試行しない。部分編集バッファを破棄し、`fail_mode=raise` なら停止、`fallback_original` なら保持したbaselineを返す。baseline再呼び出し・container再消費は禁止する。fallbackした呼び出しは `edit_applied=false` とする。

### 3.5 モード・初期設定

| 項目 | 初期値 | 用途 |
|---|---|---|
| `mode` | `observe` | 初回は本体Attentionを変えず、layoutとdispatchを確認 |
| `implementation` | `auto` | 未検証環境はmanual。検証済み条件だけaugmented |
| `strength` | `0.05` | edit時の控えめな試行値。最適値と主張しない |
| `schedule` | `curve` | 前半LF、後半HF |
| `cutoff` | `0.25` | 元実装の周波数マスクを継承 |
| `preserve_dc` | `true` | 周波数DC成分を保持 |
| `soft_mask` / `mask_width` | `true` / `0.05` | 境界の急な切り替えを抑える |
| `branch_mode` | `positive_only` | Rawでもnegative branchへの直接介入を避ける |
| `block_scope` | `0-3` | 小範囲から試す実験開始値 |
| `start_percent` / `end_percent` | `0.0` / `1.0` | 実際に渡されたdenoising schedule上の区間 |
| `zero_strength_mode` | `original` | 既定のzero strengthは完全なpassthrough |
| `spectral_diag` | `off` | 通常利用で追加のスペクトル計算をしない |
| `debug_level` | `summary` | 適用数とskip理由を確認 |
| `fail_mode` | `raise` | 診断版では異常を見逃さない |

`mode` は `off / discover / observe / edit`。`discover` は全本体blockの情報を記録し、scopeで観測候補を隠さない。`observe` は指定scopeを観測する。zero strengthは `original / manual / augmented` を指定できる。`original` はoverrideなし、後二者は指定方式を強制してalpha=1の診断対照を作る。`zero_strength_mode` はstrength=0のときだけ有効で、その場合は `implementation` より優先する。後二者を通常baselineとして使わない。`fail_mode` は `raise / fallback_original` とし、診断版はraiseを既定にする。

初版はentropy gateを無効に固定する。元実装のbeta/gammaは増幅が大きく、joint Attention用に未検証のまま継承しない。entropyを加える場合は、全Keyのentropyかテキスト内条件付きentropyかを明示した別段階の設計にする。

進行度は `sigmas` と `sample_sigmas` から得る。初期の実モデル検証はEuler/simple。多段評価・adaptive samplerなどでsigmaがschedule点に対応しなければ、最近傍に無条件で押し込まず `unsupported_progress` を記録する。1-stepでも除算エラーを起こさない。

### 3.6 ログ

JSONLはAnimaとの誤混同を避け、`model_family='krea2'`、`schema_version=4` を付ける。必須情報は以下。

- `run_id`、モデル・ComfyUI・Python・PyTorch・CUDA・GPU情報、baseline backend、編集backend、採用プロファイルID。
- `requested_implementation / selected_implementation / selection_reason`、元／拡張head dimension、元scale、FFT dtype、SDPA dtype、SDPA kernel（未確認ならunknown）。
- `step_index`、`sigma`、`progress`、同じstep内の `model_eval_index`、`block_index`。
- QKV形状、テキスト長、画像長、grid、`img_slice`、GQA展開状態、CFG layout、選択branch。
- `edit_applied`、対象scope、alpha_lf/hf、head/query chunk、見積作業領域。最大logits差は診断時だけ計算し、未計測を0としない。
- `unsupported_augmented_backend / unsupported_mask / profile_not_qualified / numerical_failure / fallback_original` の理由別件数。
- `target_skipped`、`branch_not_selected`、`unsupported_layout`、`reference_not_supported`、`memory_guard`、`unsupported_progress` などの理由別件数。

GPUメモリ測定はComfyUIサーバー内のrun wrapperで行い、API runnerはそのログを収集する。別プロセスのrunner側で測ったCUDA peakをサーバーの使用量として扱わない。時間は初回ロードを含むcold runと、モデルが用意されたwarm runを分け、測定区間と同期方法を記録する。

スペクトル診断は、画像→テキストlogitsをテキストKey内でsoftmaxした集中度のHF比を `rho_text_conditional` として記録する。これは画像の解像度やプロンプト追従性の直接測定ではない。

joint softmax中のテキストKeyの確率合計を `text_attention_mass` として別に記録する。SDPAは確率を返さないので、augmentedの観測は指定step・block・headだけ、元QKVから分割計算して求める。全headのlogitsを再生成して高速化を相殺しない。診断用作業領域も同じ予算に含め、編集scopeと観測scopeを分ける。性能比較ではspectral診断とTensor差分診断をoffにする。

### 3.7 対応範囲

| 組み合わせ | 初版の扱い |
|---|---|
| Turbo、Euler/simple、CFG=1、静止画 | 最初の実モデル検証対象 |
| Raw、Euler/simple、CFG>1 | CFG branch保持を確認する第2対象 |
| FP8モデル重み、通常のBF16/FP16 activation | 対応候補。実際のdtypeを確認 |
| INT8 ConvRotなどの重み | 対応候補。preferred Attention backendの保持を独立確認 |
| 通常のLoRA | 対応候補。AFMなしのLoRA baselineとも比較 |
| 長方形画像 | 初版で実装し、正方形に続けて検証 |
| 通常img2img | 同一token構造なら対象。実際のsigma subsetで検証 |
| reference/style-reference画像の連結 | 検出して明示的に対象外とする |
| SageAttention／INT8 Attention等 | baselineのdispatchを保持。augmentedの初版はPyTorch SDPAを使い、専用量子化kernelへの拡張QKV投入はしない |
| Q/K拡張 `d=128→256` | 数学的な設計条件。GPU・dtype・形状別の動作確認を必要とする |
| Krea2T EnhancerのAttention-Weighted Phrasesとの同時使用 | 参照元であって併用保証ではない。初版は既存override競合として拒否 |
| NAG、regional、独自QKV／Attention出力パッチ | 初版は競合検出して停止 |
| torch.compile、multi-GPU、複数フレーム | 初版の保証対象外。検出できる場合は開始前に停止 |

FP8/INT8の重みでもFFTまで低精度化しない。低精度モデルの全重みを展開する処理は追加しない。

## 4. ファイル構成と型

| 作成ファイル | 責務 |
|---|---|
| `__init__.py`、`nodes.py` | V3ノード登録、UI schema、model cloneへの設置 |
| `afm/config.py` | immutable設定と検証 |
| `afm/capabilities.py` | Krea2／ComfyUI機能・競合検出 |
| `afm/runtime.py` | run lifecycle、layout、進行度、scope、branch |
| `afm/backend.py` | 元Attention dispatch、containerの所有権 |
| `afm/spectral.py` | logits/Qで共有する空間FFT、周波数mask、DC保持 |
| `afm/joint_attention.py` | manualのhead／Query分割、joint正規化 |
| `afm/augmented_attention.py` | Q/K/V拡張、分割SDPA、head単位の出力復元 |
| `afm/selection.py`、`afm/qualified_profiles.json` | 実行方式の事前選択、検証済み条件。初期JSONは空配列 |
| `tests/reference.py` | 小規模dense reference専用。productionからimportしない |
| `afm/memory.py` | 作業領域見積もり・chunk選択 |
| `afm/diagnostics.py` | JSONL、条件付きrho、text mass |
| `tests/conftest.py`、`tests/test_*.py` | 最小限のComfyUI APIスタブ、重み・GPUなしの数値と状態分離テスト |
| `tests/gpu/test_krea2_afm.py` | 指定モデル／指定ComfyUIでのopt-in統合検証 |
| `scripts/run_afm_matrix.py`、`scripts/summarize_krea2_afm.py` | 固定条件のAPI比較と集計 |
| `examples/*.json`、`README.md`、`docs/validation.md` | 実際に開けるworkflow、設定例、対応実績 |

`afm/__init__.py` も作成する。実装単位が小さい間は一部の補助型を同じファイルに置いてよいが、Animaの約2,500行のoverrideを名前変更して複製しない。

型を先に揃える。AFMConfigはconfig.py、Krea2Capabilitiesはcapabilities.py、TokenLayout／ProgressInfo／RunStateはruntime.py、ChunkPlanはmemory.py、BackendSelectionはselection.py、AugmentedKVはaugmented_attention.py、ValidationMetricsはdiagnostics.pyに置く。

- `AFMConfig`：3.5の設定を持つfrozen dataclass。UIもテストも同じ値を使う。
- `Krea2Capabilities`：`patch_size / block_count / query_heads / kv_heads / uses_containers` と対象モデル情報。
- `TokenLayout`：`text_slice / image_slice / grid / total_tokens / reference_tokens`。sliceは `(start, end)` の整数tuple。
- `ProgressInfo`：`step_index / num_steps / progress / sigma`。
- `RunState`：run識別子、call統計、理由別件数。Tensorを永続保持しない。
- `ChunkPlan`：`head_chunk_size / query_chunk_size / estimated_workspace_mib / implementation`。
- `BackendSelection`：`implementation / reason / profile_id`。実行後に別方式を使ったのにこの値を保持することはしない。
- `AugmentedKV`：`k / v / original_head_dim`。当該head群の処理後に破棄し、runを超えてキャッシュしない。
- `ValidationMetrics`：`rel_l2 / scaled_max_abs / effect_rel_l2 / effect_resolved / finite`。6.2で定義する。

## 5. 実装タスク

各Taskで、指定された意味のあるテストを先に書いて失敗を確認し、実装して成功を確認する。コマンドは新リポジトリrootで実行する。テスト用PyTorchはCPU版でよく、ユーザーのGPU環境の依存を勝手に更新しない。

### Task 0：Krea2判定・構成・対応APIの固定

**Files:** Create `afm/config.py`, `afm/capabilities.py`, `tests/test_capabilities.py`, `README.md`, `docs/validation.md`, `afm/__init__.py`。

併せて `pyproject.toml` と `tests/conftest.py` を用意する。pytestの `gpu` markerを登録し、通常のCPUテストからGPU testsを除外する。ComfyUIに依存するテストには最小APIスタブを用い、実モデルの成功をスタブで代替しない。

**Interfaces:** `detect_krea2(model_patcher) -> Krea2Capabilities`、`validate_config(config: AFMConfig, caps: Krea2Capabilities) -> None`、`detect_patch_conflicts(model_patcher) -> list[str]`。

- [x] Animaと一般Fluxを拒否し、Krea2の構造・属性を確認する `test_reject_non_krea2` を書く。単に `.blocks` を持つだけのmockをKrea2と扱わない。
- [x] `test_scope_and_window_validation` で `start > end`、存在しないblock index、head/query chunkが0、NaN strengthを拒否する。
- [x] `python -m pytest tests/test_capabilities.py -q` で失敗を確認する。
- [x] config、モデル判定、必要APIの機能検出を実装する。新パッケージからAnimaノードをimportしない。`implementation`、zero対照、fail_modeのenumを固定し、SDPAの明示scale非対応を検出する。
- [x] 移植元・参考元のライセンスと転用箇所を記録し、必要なLICENSE／NOTICEを同梱する。
- [x] 同コマンドで全PASSを確認し、対応を確認したComfyUI commitと、まだGPUで未確認の項目を記録する。
- [x] `feat: add Krea2 AFM capability detection` としてcommitする。

### Task 1：診断用runtime・layout・baseline dispatch

**Files:** Create `afm/runtime.py`, `afm/backend.py`, `afm/diagnostics.py`, `tests/test_runtime.py`, `tests/test_backend.py`, `nodes.py`, `__init__.py`, `examples/krea2_afm_discover_api.json`, `examples/krea2_afm_observe_api.json`。

**Consumes:** Task 0のconfigとcapabilities。

**Produces:** `layout_from_latent(x: Tensor, patch_size: int, text_tokens: int) -> TokenLayout`、`validate_post_input(payload: dict, layout: TokenLayout) -> TokenLayout`、`progress_from_sigmas(options: dict) -> ProgressInfo | None`、`make_outer_sample_wrapper(config: AFMConfig) -> Callable`、`make_diffusion_wrapper(config: AFMConfig) -> Callable`、`make_post_input_patch(config: AFMConfig) -> Callable`、`dispatch_baseline(backend: Callable, q, k, v, heads: int, kwargs: dict) -> Tensor`。

この段階で `Krea2AFMAttentionOverride(config: AFMConfig)` と `install_krea2_afm(model_patcher, config: AFMConfig)` も作る。ただし `off / discover / observe` だけを利用可能とし、editは未実装エラーにする。初期診断用のV3ノードと2つのAPI workflowを公開し、Task 4で編集経路を追加する。

wrapperはComfyUIのexecutorを第1引数に取り、元の引数を維持してexecutorを呼ぶ。post_inputはpayloadを返し、その実行用 `transformer_options` にmetadataとoverrideを設置する。

- [x] `test_grid_uses_patch_padding`：latentが `[1,16,65,97]`、patch=2、text=5ならgrid `(33,49)`、画像数 `1617`、image slice `(5,1622)` である。
- [x] `test_single_frame_5d_matches_4d`：`T=1` は同じlayout、`T>1` とreference付きは対象外になる。
- [x] `test_post_input_rejects_reordered_or_extra_tokens`：row-majorでない `img_ids`、予期しないtoken増減を拒否する。`test_attention_rejects_img_slice_mismatch` では、post_inputの後に本体が設定する `img_slice` をoverride入口で検証する。
- [x] `test_preferred_backend_preserved`：module preferred backendとglobal defaultが異なるfixtureで、observeはpreferredを1回だけ呼ぶ。
- [x] `test_container_consumed_once`、`test_txtfusion_has_no_override`、`test_run_state_isolated_after_exception` を書く。後者は2つのcloneと連続runで異なるrun_idを確認する。
- [x] `python -m pytest tests/test_runtime.py tests/test_backend.py -q` で失敗を確認する。
- [x] runtime、元dispatch、discover／observeを実装する。sigma情報を全GPU Tensorのログ出力に展開しない。
- [ ] 最小の診断ノードとAPI workflowを用意する。クラウドの `/object_info` から実際のloader／sampler入力名を確認して作る。
- [x] 同コマンドで全PASSを確認する。正常終了と例外終了の両方でrun summaryを1回だけ出す。
- [x] `feat: add Krea2 AFM observe runtime` としてcommitする。

実装追記: 診断ノードとAPI workflowは作成済み。上記のクラウド `/object_info` 照合は今回未実施のため、その項目は未完了のままとする。元backend保持とcontainer一回消費は共通のパラメータ化テストで検証。

**クラウドでの中間判定:** この段階の診断ノードを試し、true baselineとobserveの再現性、txtfusion除外、block index、text/image長、grid、preferred backendを確認する。実際のログが設計と合わなければ編集実装より先にlayout／dispatchを修正する。

### Task 2：AFM周波数kernel

**Files:** Create `afm/spectral.py`, `tests/test_spectral.py`。

**Produces:** `schedule_alphas(config: AFMConfig, progress: float) -> tuple[float,float]`、`radial_low_high_masks(grid: tuple[int,int], cutoff: float, soft_mask: bool, mask_width: float, device) -> tuple[Tensor,Tensor]`、`apply_spatial_filter(x: Tensor, grid: tuple[int,int], alpha_lf: float, alpha_hf: float, config: AFMConfig) -> Tensor`、`edit_image_text_logits(logits: Tensor, grid: tuple[int,int], alpha_lf: float, alpha_hf: float, config: AFMConfig) -> Tensor`。

`apply_spatial_filter` は `[B,H,I,C]` を受け、I軸をgridへ並べ替えて共通のAを適用し、同形状のFP32を返す。logits用wrapperのCはT、augmented用のCはd。同じ周波数mask・FFT規約を共有する。

- [x] `test_curve_schedule_endpoints`：s=.1でu=0なら `(1.1,1.0)`、u=1なら `(1.0,1.1)`、u=.5なら `(1.05,1.05)` を近似比較する。
- [x] `test_fft_identity_rectangular`：grid `(3,5)`、FP32でalphaが両方1なら、入力との最大差 `<1e-5`。
- [x] `test_preserve_dc`：空間平均が元の値を `atol=1e-5, rtol=1e-5` で維持する。
- [x] `test_frequency_selectivity`：既知のLF/HF信号について、選択した周波数側だけがgainに応じて変わる。非正方形・非2の累乗のgridを含める。
- [x] `test_common_operator_linearity`：同じAについて `A(a*x+b*y)` と `a*A(x)+b*A(y)` をFP32で比較する。Qとlogitsで別のmask実装を使っていないことも確認する。
- [ ] `python -m pytest tests/test_spectral.py -q` で失敗を確認する。
- [x] 数学仕様からmask・FFT・DC処理を新規実装する（未ライセンスのソースはコピーしない）。beta/gamma付きentropy gateは入れない。
- [x] 同コマンドで全PASSを確認する。
- [ ] `feat: add spatial AFM spectral kernel` としてcommitする。

### Task 3：joint softmax・分割計算・メモリ制御

**Files:** Create `afm/joint_attention.py`, `afm/memory.py`, `tests/reference.py`, `tests/test_joint_attention.py`, `tests/test_memory.py`。

**Consumes:** Task 1のlayout、Task 2のFFT関数。

**Produces:** `estimate_full_joint_logits_mib(batch: int, heads: int, image_tokens: int, text_tokens: int) -> float`、`choose_chunks(q_shape: tuple, k_shape: tuple, layout: TokenLayout, config: AFMConfig, implementation: str, activation_dtype: torch.dtype) -> ChunkPlan`、`joint_image_rows(q: Tensor, k: Tensor, v: Tensor, layout: TokenLayout, config: AFMConfig, progress: ProgressInfo, chunks: ChunkPlan, scale: float) -> Tensor`。

`joint_image_rows` のQKVはhead形式 `[B,H,N,D]`、戻り値は `[B,H,I,D]`。上流ですでにGQA展開済みなら再展開しない。内部のlogits／softmaxはFP32、戻り値はVのdtypeにする。

- [x] `tests/reference.py` に `dense_joint_image_rows(q, k, v, layout, config, progress, scale) -> Tensor` を作る。画像→テキスト領域だけを編集して全Keyでsoftmaxする。実モデルkernelから呼び出せない配置にする。
- [x] `test_chunked_matches_dense_joint`：FP32、B=2、H=6、grid `(3,5)`、T=5、D=8で `atol=1e-5, rtol=1e-5`。head/queryの端数chunkも検証する。
- [x] `test_does_not_normalize_text_separately`：テキストと画像Keyの両方が有効なfixtureで、結果がdense jointと一致し、text-only softmaxの結果と一致しない。
- [x] `test_does_not_double_gqa`：すでに展開済みのQ/K/Vでhead数が増えない。
- [x] `test_memory_dimensions`：B=1,H=48,I=4096,T=512の全joint logits見積もりは `3888.0 MiB`。作業予算を超える場合、FFT前にchunk縮小または明示的に停止する。
- [x] `test_failure_does_not_return_partial_edit`：chunk途中で例外／NaNを注入し、部分編集を返さない。
- [ ] `python -m pytest tests/test_joint_attention.py tests/test_memory.py -q` で失敗を確認する。
- [x] head分割とQuery行分割を実装し、FFTのQuery軸は全gridを維持する。mask非None・causal条件は初版で編集せず理由を返す。
- [x] 同コマンドで全PASSを確認する。密な全joint配列をproduction pathで確保しないことをコードレビューする。
- [ ] `feat: add chunked Krea2 joint AFM attention` としてcommitする。

### Task 3A：Q／K拡張kernel・方式選択

**Files:** Create `afm/augmented_attention.py`, `afm/selection.py`, `afm/qualified_profiles.json`, `tests/test_augmented_attention.py`, `tests/test_selection.py`。Modify `afm/memory.py`, `tests/test_memory.py`。

**Consumes:** Task 1のlayout、Task 2の共通FFT、Task 3のmanualとdense reference。通常のSDPA入力はGQA展開後のhead形式を前提とする。

**Produces:**

- `build_augmented_kv(k: Tensor, v: Tensor, layout: TokenLayout) -> AugmentedKV`：入力は当該head群の全Key／Value。
- `augmented_image_rows(q: Tensor, k: Tensor, v: Tensor, layout: TokenLayout, config: AFMConfig, progress: ProgressInfo, chunks: ChunkPlan, scale: float) -> Tensor`：manualと同じ `[B,H,I,d]`／V dtypeを返す。
- `strip_augmented_heads(out: Tensor, original_head_dim: int, heads: int, merged: bool) -> Tensor`：head形式／merged形式ともhead境界を保持する。
- `select_implementation(config: AFMConfig, runtime_signature: dict, profiles: list[dict]) -> BackendSelection`：pureな事前選択。runtime_signatureは3.4.4の環境情報と実際の形状を含む。

- [x] `test_augmented_logits_match_direct_fft`：B=2,H=3,grid=(3,5),T=4,d=8、alpha=(1.08,1.14)で、Q/K拡張の内積と直接FFTしたlogitsが `atol=1e-5,rtol=1e-5` で一致する。画像Key列のlogitsは同許容誤差内で元と一致する。
- [x] `test_augmented_output_matches_dense_and_manual`：全Key SDPA出力が両referenceに同許容誤差で一致し、text-only softmaxとは一致しない。head/query端数chunkを含める。
- [x] `test_original_effective_scale_preserved`：元の明示scale=0.2と未指定相当のd**-0.5を試し、拡張dimからの再計算を検出する。
- [x] `test_strip_per_head_not_flat_prefix`：H=3,d=8でhead別に異なる値を使い、両出力形式で正しいhead順・値が復元される。追加V次元はゼロである。
- [x] `test_augmented_zero_is_numerical_control`：alpha=1でも拡張経路を通ったことを記録し、zero originalの完全passthroughと区別する。
- [x] `test_fft_before_query_chunking`：Query chunk幅が変わっても同じ全grid FFTを使う。`test_sdpa_call_is_bounded`：全callが設定head/query上限以下、Key長は常にN。
- [x] `test_mask_causal_and_unsupported_scale_rejected`：mask非None、causal、scaleを渡せないbackendを黙って受け入れない。
- [x] `test_auto_unqualified_uses_manual`、`test_auto_profile_scope_exact`：空profile、dtype/GPU/ComfyUI変更、検証範囲外のT/I/batchではmanual、合格profile内だけaugmented。forced augmentedの未対応はエラー。
- [x] `test_augmented_memory_counts_math_and_retained_qkv`：mathのFP32中間配列とbaseline後まで保持するQKVを含める。表の4 MiB／4.5 MiBも検算する。
- [ ] `python -m pytest tests/test_augmented_attention.py tests/test_selection.py tests/test_memory.py -q` で失敗を確認する。
- [x] 共通FFTと元scaleを使って実装する。DeltaQはFP32で求め、SDPA直前だけactivation dtypeへ変換する。`qualified_profiles.json` の初期値は `[]`。
- [x] 同コマンドで全PASSを確認する。CPU SDPA成功をGPU fused対応と記載しない。
- [ ] `feat: add augmented QK AFM candidate and guarded selection` としてcommitする。

### Task 4：ノード統合・branch・競合・passthrough

**Files:** Create `tests/test_node_patch.py`, `tests/test_branch_preservation.py`。Modify `nodes.py`, `__init__.py`, `afm/runtime.py`, `afm/backend.py`。

**Consumes:** Tasks 0–3Aの設定、runtime、backend、2つのkernelと方式選択。

**Produces:** `install_krea2_afm(model_patcher, config: AFMConfig)`。`Krea2AFMModelPatch` はMODELを受け取り、パッチ済みMODELを返す。

- [x] `test_zero_original_installs_no_attention_override`：off／既定のzero strengthで元のAttention経路が保持される。
- [x] `test_positive_only_layouts`：`[0]`、`[1]`、`[1,0]`、`[0,1]`、batch>1、metadata欠落を検証する。metadata欠落をpositiveと決めつけない。
- [x] `test_unselected_branch_and_text_rows_bitwise_equal`：manual／augmentedの両方で、同じ呼び出し内のbaselineと出力を `torch.equal` で比較する。未選択branchとテキストQuery行を保持する。
- [x] `test_input_model_not_modified`：同じloaderからAFM modelと通常modelに分岐し、通常modelの設定／出力を保持する。
- [x] `test_duplicate_afm_rejected`、`test_existing_attention_patch_rejected`：二重AFM、別override、post_input/QKV/outputパッチを明示的なエラーにする。
- [x] `test_fallback_reuses_baseline_once`：両kernelの途中chunkにOOM／NaNを注入し、fallback_originalでは保持済みbaselineを返して部分編集を破棄する。元backendは1回だけ、containerは1回だけ消費される。raiseでは例外を再送出し、別kernelを無断で再試行しない。
- [x] `test_zero_control_precedence`：strength=0のmanual／augmented指定はimplementationより優先し、originalだけがoverrideなしになる。
- [x] `test_normal_augmented_path_has_no_dense_diagnostics`：診断offではlogits再生成を呼ばない。診断onでも指定head/queryの上限を守る。
- [ ] `python -m pytest tests/test_node_patch.py tests/test_branch_preservation.py -q` で失敗を確認する。
- [x] V3 schemaと設置処理を実装する。Animaのscope名 `encoder_equivalent` と、Anima固有のcall index推定を持ち込まない。
- [x] 同コマンドで全PASSを確認し、`python -m pytest tests -q` と `python -m compileall -q afm nodes.py __init__.py` を実行する。GPU testsはopt-in markerで通常CPU実行から除外する。
- [ ] `feat: expose Krea2 AFM model patch node` としてcommitする。

### Task 5：実モデル比較・workflow・実験report

**Files:** Create `tests/gpu/test_krea2_afm.py`, `scripts/run_afm_matrix.py`, `scripts/summarize_krea2_afm.py`, `examples/validation_matrix.json`, `examples/krea2_afm_edit_api.json`, `examples/krea2_afm_edit_ui.json`, `tests/test_reports.py`, `tests/test_validation_metrics.py`, `scripts/compare_attention_kernels.py`。Modify `examples/krea2_afm_discover_api.json`, `examples/krea2_afm_observe_api.json`, `README.md`, `docs/validation.md`, `afm/diagnostics.py`, `afm/qualified_profiles.json`。

**Produces:** 実際にComfyUIに読み込めるworkflow、固定条件run manifest、画像／JSONL／集計CSV、実測対応表、採用候補profile。

**Interface:** `compare_attention_outputs(candidate: Tensor, reference: Tensor, candidate_zero: Tensor, reference_zero: Tensor) -> ValidationMetrics` を `scripts/compare_attention_kernels.py` に置き、6.2の定義で計算する。実モデルではサーバー内の診断hookが同じQKVから比較し、API runnerへ数値だけ渡す。外部runnerからサーバーのTensorへアクセスできると仮定しない。

- [x] `test_summary_requires_edited_calls`：画像が保存されてもAFMの `edited_calls=0` なら「編集成功」にしない。
- [x] `test_compare_pairs_are_unambiguous`：run間比較keyを `step_index / model_eval_index / block_index / branch` とし、重複・欠損を黙って平均しない。
- [x] `test_numeric_metrics_and_zero_effect`：既知のTensorで6.2の指標を検算し、効果差が数値ノイズ以下なら合格ではなく判定不能にする。
- [x] `test_profile_requires_correctness_and_benchmark`：画像保存だけ、全fallback、性能未測定、数値閾値外のreportから採用profileを作れない。
- [ ] `python -m pytest tests/test_reports.py tests/test_validation_metrics.py -q` で失敗を確認し、集計を実装してPASSを確認する。
- [ ] cloudの `/object_info` と実際の標準workflowからinputsを確認し、API JSONとUI JSONを作る。存在しないノード名やsocketを推測で使わない。
- [x] モデル名やComfyUI URLを引数で指定するrunnerを作る。実行例は `python scripts/run_afm_matrix.py --comfy-url http://127.0.0.1:8188 --matrix examples/validation_matrix.json --out-dir outputs/krea2_afm`。この `examples/validation_matrix.json` も作成する。
- [ ] 6節の順でクラウドGPU検証を実行する。診断版branchは `diagnostic/krea2-afm`、編集版は `feat/krea2-afm-edit`、Q/K拡張の追加試験は `feat/krea2-afm-augmented` を候補とする。
- [ ] 同一QKVのFP32／FP16／BF16差分を6.2で測る。続いて6.3のwarm比較を実施し、採用条件に合格したprofileだけ `afm/qualified_profiles.json` に追加する。自動登録で受入閾値を緩めない。
- [ ] batch=1、まず小解像度、次に768正方形、長方形、可能なら1024で測定する。`torch.cuda.max_memory_allocated()`、reserved memory、wall timeをmanifestに入れる。
- [x] 未確認の量子化／backendを対応済みにしない。実績のある設定だけをREADMEの初期例にする。
- [ ] `docs: add Krea2 AFM workflows and validation results` としてcommitする。


再開後の記録: Task 2–4とTask 5のオフライン実装/CPU検証を完了。未チェックのRED履歴はこの再開で再実施しておらず、実モデル項目は対象外のため未実施。タスク別commit名は原案であり、編集実装はfeat/krea2-afm-edit上の1つのローカルcheckpointへまとめる。API/UI例は静的確認のみ。

## 6. クラウドGPUでの検証順

| 段階 | 設定 | 合格条件 |
|---|---|---|
| A0/A1 | AFMノードなし、同条件を2回 | baseline自身の再現性を把握 |
| B | discover、スペクトル診断off | 本体block・gridを確認。txtfusionに介入しない |
| C | observe、必要blockだけ診断 | 元backendを維持。再現性の範囲でbaselineに一致。edited=0 |
| D | edit、strength=0、zero=original | overrideなし。baselineと同じ経路 |
| E1/E2 | edit、strength=0、zero=manual／augmented | 各方式の数値差を測定。通常baselineには使わない |
| F1 | manual、strength=.05、block=0 | 非zero編集、全Key正規化、NaNなし、未選択行保持 |
| F2 | 同一QKVをmanual／augmentedで計算 | 6.2の数値基準。head=128→256と元scaleも確認 |
| F3 | augmented、strength=.05、block=0 | 実生成、fallbackなし、時間・VRAM・生成画像を確認 |
| G | .05/.1/.2、小block範囲 | manualで効果を評価し、合格augmentedと比較 |
| H | Raw、CFG>1、positive_only | batch／別forward双方でbranch保持を確認 |
| I | 長方形、量子化形式、LoRA、img2img | 各対象を個別に判定・記録 |
| J | manual／augmentedのwarm比較 | 6.3を満たした条件のみauto採用profileへ追加 |

別runの最終画像だけでkernelの等価性を評価しない。各runでは後続block・stepでQKV自体が分岐するため、F2は同一のQKV入力に対する比較が必要である。非決定的backendについてはA0/A1の変動と同一call内の保持テストを併記する。

### 6.1 効果の評価セット

最初のpilotは8 prompts × 4 seeds。プロンプトは次の8テーマで固定し、比較途中に改変しない。

1. 異なる色の2物体と左右関係。
2. 個数を指定した単純物体。
3. 人物の衣服・髪・所持品の属性対応。
4. 全身人物のポーズと背景。
5. 布・レースの微細模様。
6. ガラス・金属・反射。
7. 光源と複雑な背景。
8. 短い指定文字列を含む看板。

まず1つのblock範囲で `.05/.1/.2` を比較し、有望なstrengthを固定してから範囲を比較する。全strength・全範囲・全seedを最初から総当たりしない。

28 blocksの確認後の範囲候補は early `0-6`、middle `10-14`、late `21-27`。層の位置に基づく実験候補であり、効果のある層と確認されたものではない。

各画像について、要求した対象・色・属性・左右・個数の充足率、細部の見やすさ、破綻／過剰模様、構図や人物の変化を分けて評価する。同じseed・prompt・latent・sampler・steps・CFG・LoRA strength・backendを維持する。

HF比上昇やシャープな見た目だけでは「プロンプト追従性向上」と判定しない。pilot結果は効果の傾向として報告し、改善率を一般化しない。

### 6.2 数値差の受入基準

CPUではTask 3／3Aの小規模FP32 fixtureを `atol=1e-5, rtol=1e-5` で比較する。GPUでは実際のpost-RoPE QKVを固定し、選んだstep・block・branch・headについて同じ入力を両kernelへ渡す。logits比較は小さいhead群・Query chunkのみ、出力比較は選択画像行で行う。manual FP32 referenceは、同じ実測QKVをFP32へcastして計算し、出力もFP32で保持する。モデル自体をFP32で再実行して別のQKVを作らない。

```text
rel_l2 = ||candidate-reference||_2 / max(||reference||_2, 1e-6)
scaled_max_abs = max(abs(candidate-reference)) / max(max(abs(reference)), 1.0)
reference_effect = reference_s - reference_zero
candidate_effect = candidate_s - candidate_zero
effect_rel_l2 = ||candidate_effect-reference_effect||_2 / max(||reference_effect||_2, 1e-6)
```

以下は初版の採用判定値であり、実測済み精度を示す値ではない。高精度比較でも実行環境と行列積設定を記録する。

| 比較 | rel_l2上限 | scaled_max_abs上限 |
|---|---:|---:|
| augmented FP32 vs manual FP32 | 1e-4 | 1e-4 |
| augmented FP16 vs manual FP32 | 5e-3 | 1e-2 |
| augmented BF16 vs manual FP32 | 2e-2 | 5e-2 |

すべて有限値で、テキスト行・未選択branchは同一callのbaselineと `torch.equal` が必要。通常のactivation dtypeでは、さらに `.05/.1/.2` で `effect_rel_l2 <= 0.25` を要求する。`||reference_effect||_2 <= max(1e-6, 3*||candidate_zero-reference_zero||_2)` なら、編集効果が方式間ノイズから十分に分離できないため「判定不能」とし、その条件はautoへ採用しない。閾値を超えたらFP32経路、DeltaQの丸め、元scale、shape復元を調べる。テストを通す目的だけで閾値を緩めない。

実モデルの全blockを両経路で二重実行する必要はない。最初は1つの代表blockと序盤／終盤stepから始め、auto採用対象を広げる際に測定対象も広げる。FP32候補でのみ合格した構成を、FP16/BF16で合格したと扱わない。

### 6.3 性能測定とautoへの採用条件

同じモデル・prompt・seed・latent・steps・scope・strength・dtypeで比較する。cold runは分離し、各方式1回のwarm-up後に5回測定する。測定順を交互にし、各run前後をサーバー内でCUDA同期する。時間は全生成時間とAttention編集区間を分け、peak allocated／reservedをサーバープロセスで記録する。比較診断・spectral診断はoff。

最低限、小解像度、768正方形、768を一辺とする長方形を測る。1024はbaselineが動く環境で追加する。Tの実測値を記録し、短い／長いpromptの両方を含める。d=128→256は全head一括実行を許可する根拠にしない。

autoへの初期採用条件は、対象範囲で数値基準合格・fallbackゼロ・NaN/OOMゼロに加え、manual比でwarm生成時間中央値とpeak allocated中央値がともに悪化せず、少なくとも片方が5%以上改善すること。5%は初期運用上の判定値であり、性能予測ではない。測定のばらつきが差より大きければ採用を保留する。条件を満たさなくてもmanual経路で初版を完成でき、augmentedは明示選択の実験経路として残す。

profileには測定manifestと数値reportの参照を付け、ComfyUI／PyTorch／CUDA更新やGPU・dtype変更時は再検証する。対応表では「manual動作確認」「augmented数値確認」「augmented性能採用」を別欄にする。

## 7. 完了判定と後続候補

初版の完了には以下を必要とする。

- [x] CPUのdense／manual／augmented比較、DC／長方形／明示scale／head単位出力復元／scope／branch／container／状態分離テストがPASS。
- [ ] 実際のKrea2 Turboでdiscover／observe／zero original／編集の4経路を確認。
- [ ] 少なくとも1種類の長方形入力と、RawのCFG branch試験を確認。
- [ ] 実ログで適用block・画像範囲・編集差を確認し、全fallbackで動いている状態を成功と判定しない。
- [x] 対応した量子化形式とbackend、未確認の形式を分けた表がある（実機確認済みはなし。docs/validation.md参照）。
- [ ] 実測VRAM・実行時間と、baseline／manual zero／augmented zero／両AFM方式の比較がある。
- [x] autoの未検証環境はmanual。augmentedの採用有無と数値／性能の根拠を明記する。未達の候補を標準高速化済みとしない。
- [ ] API/UI workflowを実際にロードして実行できる。

後続候補はentropy適応、reference画像対応、画像→画像領域の周波数編集、baseline全体計算の削減、専用backend対応／融合kernelによる高速化。entropyやtoken別gainを追加する場合は、Q/K拡張の等価性を再検討する。これらは初版の検証結果を踏まえ、別タスクとして設計する。

## 8. この計画作成時に確認したこと

前版ではAnima AFM、ComfyUIのKrea2本体、Attention dispatch、ModelPatcher、wrapper、samplerを確認した。今回、Krea2T Enhancerの `attention_weighted_phrase.py` と関連layout処理を確認し、全Key正規化・追加次元・元scale保持を設計に取り入れた。SDPAのscale、kernel選択、数値差はPyTorch公式資料を参照した。[S1, S2]

すでに実行した小規模NumPy計算は以下。PyTorch／GPUテストの代用ではない。

| 計算 | 条件・結果 |
|---|---|
| 前版のdense vs head/Query分割 | B=2,H=6,grid=(3,4),T=5,d=8。最大絶対差0、未選択branch・テキスト行も一致 |
| phrase biasの追加次元表現 | float64。直接加算との差 4.44e-16 |
| AFM Q/K拡張 vs 直接FFT logits | B=2,H=3,grid=(3,5),T=4,d=8、float64。最大絶対差 1.78e-15 |
| 同Attention出力 | 最大絶対差 1.17e-15 |
| 画像Key側のlogits | 元との差0 |

これらは共通線形フィルタの代数確認である。実モデルのFP16/BF16誤差、d=128→256のbackend制約、GPUでの速度／VRAM、生成品質の改善は未確認。6.2の閾値と6.3の5%条件は今回定めた実装上の受入基準で、観測結果ではない。

## 9. 原典・コード参照

文中の[S1]・[S2]は以下。固定commitのコードを根拠にし、実装開始時には実環境の差分を確認する。

- [S1] [Krea2T Enhancer・Attention-Weighted Phrases・固定revision](https://github.com/capitan01R/ComfyUI-Krea2T-Enhancer/blob/e1f60f79e168c62b1cb1cf773dfb3ae4ee2292e2/attention_weighted_phrase.py)
- [Krea2T Enhancer・layout・固定revision](https://github.com/capitan01R/ComfyUI-Krea2T-Enhancer/blob/e1f60f79e168c62b1cb1cf773dfb3ae4ee2292e2/_attention_weighting_layout.py)
- [Krea2T Enhancerリポジトリ](https://github.com/capitan01R/ComfyUI-Krea2T-Enhancer)
- [S2] [PyTorch SDPA公式資料](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html) — 参照資料の版であり、PyTorch 2.14を最低要件に指定するものではない。

- [移植元リポジトリ](https://github.com/Shiba-2-shiba/AFM-for-anima-experimental)
- [移植元AFM本体・固定revision](https://github.com/Shiba-2-shiba/AFM-for-anima-experimental/blob/596abc87f4db983aaceea19b437119367f45eb23/anima_afm.py)
- [移植元ノード定義・固定revision](https://github.com/Shiba-2-shiba/AFM-for-anima-experimental/blob/596abc87f4db983aaceea19b437119367f45eb23/nodes.py)
- [ComfyUI Krea2モデル・固定revision](https://github.com/Comfy-Org/ComfyUI/blob/9d80841aa1990305cc7a280c5ea505f317efcc2d/comfy/ldm/krea2/model.py)
- [ComfyUI Attention API・固定revision](https://github.com/Comfy-Org/ComfyUI/blob/9d80841aa1990305cc7a280c5ea505f317efcc2d/comfy/ldm/modules/attention.py)
- [ComfyUI ModelPatcher・固定revision](https://github.com/Comfy-Org/ComfyUI/blob/9d80841aa1990305cc7a280c5ea505f317efcc2d/comfy/model_patcher.py)
- [ComfyUI wrapper API・固定revision](https://github.com/Comfy-Org/ComfyUI/blob/9d80841aa1990305cc7a280c5ea505f317efcc2d/comfy/patcher_extension.py)
- [ComfyUI samplers・固定revision](https://github.com/Comfy-Org/ComfyUI/blob/9d80841aa1990305cc7a280c5ea505f317efcc2d/comfy/samplers.py)
- [ComfyUI公式Krea2 Turbo workflow](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/image_krea2_turbo_t2i.json)
- [ComfyUI用Krea2モデル](https://huggingface.co/Comfy-Org/Krea-2)
- [Krea2 Raw公式model card](https://huggingface.co/krea/Krea-2-Raw)
- [AFM原論文・arXiv:2603.28114v1](https://arxiv.org/abs/2603.28114v1)

## 10. Codexに渡す開始指示

> 本書を読み、独立した `ComfyUI-Krea2-AFM` を実装してください。最初はTask 0・1の診断版を作り、既存Animaノードの挙動は変更しないでください。ローカルでCPUテストを行い、クラウドGPUへ渡せるdiscover／observe workflowとJSONL診断を用意してください。Krea2本体forwardのコピーやグローバルmonkey patchは避け、txtfusionを除外し、画像grid・preferred backend・containerの消費回数・CFG branchを確認できるようにしてください。
>
> 診断版のGPUログが前提と一致したら、Task 2・3の共通FFTとmanual基準kernelを実装し、続けてTask 3AのQ/K拡張候補を追加してください。画像→テキストlogitsだけを編集し、画像＋テキスト全Keyでsoftmaxする仕様を両方式で維持してください。QのFFTはQK norm／RoPE後、全画像gridで行います。Q/K/V拡張後も元の実効scaleを明示し、SDPA出力はhead単位で追加次元を除いてください。SDPAを使ってもhead／Query分割とmath経路のメモリ見積もりを残してください。
>
> baselineを1回だけ計算し、選択branchの画像Query行だけを置換してください。元のテキスト行・未選択branchを保持し、途中失敗で部分編集を返さないでください。通常のzero strengthはoverrideを設置せず、manual zero／augmented zeroは診断対照として別に扱ってください。
>
> autoの初期採用リストは空にし、未検証環境ではmanualを使ってください。Task 5の同一QKV比較・実生成・warm性能測定を通過した条件だけQ/K拡張をautoへ採用してください。ローカルでGPUを実行できなければ未検証を明示し、必要なworkflow・テストコマンド・ログ項目を揃えてクラウド検証へ引き継いでください。NumPyの等価性確認だけでGPU対応・高速化・画質向上を宣言しないでください。
