首先說明 ics43434 的使用方法 : 
住要pin腳 
 
BCLK        : 由master傳送給 slave 來做同步時鐘
L/RCL (WS)  : 對應到 BCLK (設定的sample rate 會等同於 WS的頻率, 設定的 bitwidth 會影響到 一個 WS 週期會包含多少 BCLK 週期)
DOUT        : 資料輸出 (會跟 W/S作對應同步, WS高電位 右聲道 ； WS低電位 左聲道) ==> 專案選左聲道
L/R (SEL)   : 選擇左右聲道就對這Pin 接 GND/3V3


程式碼運作說明 : 




debug經歷
