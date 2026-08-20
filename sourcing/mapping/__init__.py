# 待GINO改寫
"""映射：把各家的原生標籤收斂成我們要的五類，以及驗證這些映射的工具。

**這個資料夾全部不連網、不吃 GPU、跑起來都是幾秒鐘的事。** 想理解整條線
在判什麼、為什麼這樣判，從這裡開始最快。

────────────────────────────────────────────────────────────────────────
四套標籤體系
────────────────────────────────────────────────────────────────────────

  文字   BERT 8 類（平淡語氣／關切語調／開心語調／憤怒語調／悲傷語調／
                    疑問語調／驚奇語調／厭惡語調）
  臉部   AffectNet 8 類（Neutral/Happiness/Sadness/Anger/Surprise/
                         Fear/Disgust/Contempt）
  演員   RAVDESS 檔名裡的數字代號 01–08
  我們   五類：中性／喜／怒／哀／樂

各家的原生類別不必也不該互相對齊 —— AffectNet 的 Contempt 在 BERT 那邊
沒有對應，BERT 的關切語調在 AffectNet 那邊也沒有。三套各自收斂到五類，
**只在五類這一層比較**。值為 None 一律代表棄權（不硬塞）。

────────────────────────────────────────────────────────────────────────
檔案
────────────────────────────────────────────────────────────────────────

  schemes.py        ★ 三張映射表都在這裡（第四張 NATIVE_TO_FIVE 在
                    emotion/labels.py，因為上線會用到，這裡 re-export）。
                    載入時就斷言值域都落在五類之內 —— 打錯字當場炸掉，
                    而不是跑完幾百支影片才發現某一類永遠是 0。
                    只放對照關係，不做任何運算。

  show_mapping.py   把四套體系攤開來印，含反向表（每個五類是從哪些原生類別
                    來的）。要確認「怒到底吃進了什麼」跑這支。
                      python sourcing/mapping/show_mapping.py
                      python sourcing/mapping/show_mapping.py --reverse

  crosstab.py       映射表的**實證依據**：拿 eval 跑完的 predictions.jsonl
                    重算「模型判這一類時，真實標籤實際上是什麼」。
                    不重跑模型，換映射／換門檻／換模型重跑這支就有新數字。
                      python sourcing/mapping/crosstab.py
                      python sourcing/mapping/crosstab.py --five
                      python sourcing/mapping/crosstab.py --valence-gate

  compare.py        同一支影片，文字判的 vs 臉部判的，並排對照。
                    「文字判不出表情」這個結論的量化版本。
                      python sourcing/mapping/compare.py --uid bili:BV17z4y117cR
                      python sourcing/mapping/compare.py --all --matrix

────────────────────────────────────────────────────────────────────────
五類裡的「喜」目前產不出來
────────────────────────────────────────────────────────────────────────

喜與樂照字義分：喜偏遇事而生的欣喜（突發、反應性），樂偏處之而安的快樂
（持續、狀態性）。但這條界線在文字層畫不出來 —— 「驚奇語調」實測抓到的是
價性中立的**驚訝**（surprise 66.7%、happy 只有 6.7%），試過用 raw_probs
做價性閘門也切不開（跑 crosstab.py --valence-gate 可以自己確認）。

所以文字層只收斂到「驚」，喜／哀 的分家留給臉部表情或上下文裁決 ——
驚喜（挑眉＋嘴角上揚）與驚嚇（挑眉＋嘴部緊繃）在畫面上明顯可分。
細節見 emotion/labels.py 的註解。
"""
