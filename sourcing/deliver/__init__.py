# 待GINO改寫
"""交付物：從「哪幾支有希望」到「可以直接丟進 JoyGen 的影片檔」。

交出去的不能只是「第 42 秒到第 47 秒」這種清單 —— 對方還得自己下載、
自己切。所以最後一步是實際產出檔案，外加一份 manifest 說明
每個檔案是誰的哪一種情緒。

────────────────────────────────────────────────────────────────────────
檔案
────────────────────────────────────────────────────────────────────────

  shortlist.py        把候選影片排出優先順序，決定接下來要跑哪幾支 detect/。
                      已跑過 detect 的用實測數字（non_neutral_ratio、各類片段數），
                      還沒跑過的用 metadata 估分。估分的三個訊號，
                      實測有效性由高到低：
                        1. 題材 —— 人生經歷型敘事（離職／分手／霸凌／失敗）
                           非中性佔比 40–50%，平穩經驗分享型口播只有 20%
                        2. 時長 —— 20 分鐘以上命中率明顯較高
                        3. 性別 —— 最終要交女性角色，女性訊號加分
                           （只是關鍵字猜測，要看封面才算確認）
                        --top 30 / --female-only / --min-duration 600
                        --pick 12   直接印出可貼給 detect/ 的 --uid 參數

  make_review_page.py 產出單一 HTML 的人工檢視頁，封面圖用 data URI 內嵌，
                      丟到哪裡都能開。每個切點做成
                      `bilibili.com/video/BVxxx?t=秒數` 的連結，
                      **點下去直接跳到那一秒**，不必自己拉進度條。
                        --min-labels 4 --out review.html

  cut_clips.py        用 yt-dlp 下載 + ffmpeg 切段，產出實際的 mp4 與 manifest.json。
                      只吃 face_timeline.py 的結果，不吃文字那條。
                        --uid bili:BV1aM4y117kD
                        --all --per-emotion 3
                        --height 720 --pad 0.3 --min-seconds 3.0

                      兩個實作上的決定：
                      - **重新編碼，不用 -c copy**。copy 只能從關鍵影格切，
                        實際起點會飄到好幾秒前，切出來可能根本不是那個表情。
                      - **前後各留 pad 秒**。FER 是逐張判定的，表情起訖點
                        會被切得很死，留一點過渡對嘴型模型也比較好處理。

  演員素材那條不在這裡 —— 它從解壓到分類一支就做完，見 sourcing/actors/。

────────────────────────────────────────────────────────────────────────
輸出
────────────────────────────────────────────────────────────────────────

  _local/sourcing/shortlist.jsonl         候選排序
  _local/sourcing/review.html             人工檢視頁
  _local/sourcing/material_table.json     素材表（人 → 情緒 → 切點）
  _local/sourcing/clips/{作者}/{情緒}/*.mp4   實際片段 + manifest.json
"""
