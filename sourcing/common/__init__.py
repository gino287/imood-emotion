# 待GINO改寫
"""共用基礎：檔案位置、jsonl 讀寫、續跑、節流、終端機編碼。

這個資料夾沒有可以直接執行的腳本，只被其他資料夾 import。

  paths.py   所有輸入輸出的位置。**想知道「某一步的結果存在哪」看這個檔案就好。**
             常數依流程順序排（collect → detect → deliver），
             另有三個小函式：raw_path()／timeline_path()／face_path()。

  store.py   抓取階段共用的存檔與續跑邏輯：
               enable_utf8_stdout()  每支 CLI 的 main() 第一行都要呼叫，
                                     不然 Windows 的 cp950 終端機碰到簡體字會整支崩掉
               read_jsonl / write_jsonl     jsonl 讀寫
               load_sources()               讀 L0 的清單，缺檔時給明確指示
               has_raw / save_raw / load_raw / iter_raw   raw 層的存取（續跑靠這個）
               merge_metadata()             以 uid 為主鍵併回 metadata.jsonl，不會重複列
               append_error()               失敗紀錄
               Throttle                     兩次請求之間的最小間隔（避免被風控）

存檔刻意分成 raw/ 與 metadata.jsonl 兩層：抓取很貴（1188 支要跑二十幾分鐘）、
抽取很便宜。留著 raw，之後想多留一個欄位只要 --rebuild，不必再碰網路。
"""
