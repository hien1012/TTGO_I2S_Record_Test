#include <WiFi.h>
#include <WebServer.h>
#include <WebSocketsServer.h>
#include <driver/i2s.h>

// ========== WiFi 設定 ==========
const char* ssid = "草莓的家";
const char* password = "10122001";

// ========== I2S 腳位設定 ==========
#define I2S_SCK   26    // Serial Clock (BCLK)
#define I2S_WS    22    // Word Select (LRCLK)
#define I2S_SD    21    // Serial Data (DOUT)
#define I2S_PORT  I2S_NUM_0

// ========== 音訊參數 ==========
#define SAMPLE_RATE   16000   // 取樣率 16kHz（人聲足夠）
#define BUFFER_LEN    1024    // 每次讀取的樣本數

// 緩衝區
int32_t rawSamples[BUFFER_LEN];     // I2S 讀出的原始 32-bit 資料
int16_t samples16[BUFFER_LEN];      // 轉換後的 16-bit 資料

// 伺服器
WebServer server(80);               // HTTP 伺服器（埠 80）
WebSocketsServer webSocket(81);     // WebSocket 伺服器（埠 81）

bool clientConnected = false;       // 是否有客戶端連線

// ========== 網頁 HTML ==========
const char indexHtml[] PROGMEM = R"rawliteral(
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>ESP32 即時音訊串流</title>
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #1a1a2e;
      color: #eee;
      display: flex;
      justify-content: center;
      align-items: center;
      min-height: 100vh;
    }
    .container {
      text-align: center;
      padding: 2rem;
    }
    h1 {
      font-size: 1.5rem;
      margin-bottom: 1rem;
      color: #e94560;
    }
    .status {
      font-size: 1rem;
      margin-bottom: 1.5rem;
      color: #aaa;
    }
    .status.connected { color: #4ecca3; }
    .status.error { color: #e94560; }
    button {
      padding: 1rem 2.5rem;
      font-size: 1.1rem;
      border: none;
      border-radius: 50px;
      cursor: pointer;
      transition: all 0.3s;
      margin: 0.5rem;
    }
    #btnStart {
      background: #4ecca3;
      color: #1a1a2e;
    }
    #btnStart:hover { background: #3db890; }
    #btnStop {
      background: #e94560;
      color: #fff;
      display: none;
    }
    #btnStop:hover { background: #d63851; }
    .volume-bar {
      width: 300px;
      height: 20px;
      background: #333;
      border-radius: 10px;
      margin: 1.5rem auto;
      overflow: hidden;
    }
    .volume-fill {
      height: 100%;
      width: 0%;
      background: linear-gradient(90deg, #4ecca3, #e94560);
      transition: width 0.1s;
      border-radius: 10px;
    }
    .info {
      font-size: 0.8rem;
      color: #666;
      margin-top: 2rem;
    }
  </style>
</head>
<body>
  <div class="container">
    <h1>🎙️ ESP32 即時音訊串流</h1>
    <div id="status" class="status">點擊按鈕開始收聽</div>
    <div class="volume-bar">
      <div id="volumeFill" class="volume-fill"></div>
    </div>
    <button id="btnStart" onclick="startAudio()">▶ 開始收聽</button>
    <button id="btnStop" onclick="stopAudio()">⏹ 停止</button>
    <div class="info">
      取樣率：16kHz ｜ 位元深度：16-bit ｜ 單聲道｜ 傑森創工
    </div>
  </div>

  <script>
    let ws = null;
    let audioCtx = null;
    let isPlaying = false;
    const SAMPLE_RATE = 16000;

    function startAudio() {
      // 建立 AudioContext（瀏覽器要求必須在使用者操作後才能建立）
      audioCtx = new (window.AudioContext || window.webkitAudioContext)({
        sampleRate: SAMPLE_RATE
      });

      // 建立 WebSocket 連線
      // 自動使用目前頁面的 IP，不用手動填
      const host = window.location.hostname;
      ws = new WebSocket(`ws://${host}:81`);
      ws.binaryType = 'arraybuffer';

      ws.onopen = () => {
        isPlaying = true;
        document.getElementById('status').textContent = '🟢 已連線，收聽中...';
        document.getElementById('status').className = 'status connected';
        document.getElementById('btnStart').style.display = 'none';
        document.getElementById('btnStop').style.display = 'inline-block';
      };

      ws.onmessage = (event) => {
        if (!isPlaying || !audioCtx) return;

        // 將收到的二進位資料轉成 16-bit 整數陣列
        const int16Array = new Int16Array(event.data);
        const numSamples = int16Array.length;

        // 建立 AudioBuffer
        const audioBuffer = audioCtx.createBuffer(1, numSamples, SAMPLE_RATE);
        const channelData = audioBuffer.getChannelData(0);

        // 將 int16 轉成 float（-1.0 ~ 1.0）
        let sumSquares = 0;
        for (let i = 0; i < numSamples; i++) {
          channelData[i] = int16Array[i] / 32768.0;
          sumSquares += channelData[i] * channelData[i];
        }

        // 計算音量（RMS）並更新音量條
        const rms = Math.sqrt(sumSquares / numSamples);
        const volumePercent = Math.min(rms * 500, 100); // 放大顯示
        document.getElementById('volumeFill').style.width = volumePercent + '%';

        // 播放音訊
        const source = audioCtx.createBufferSource();
        source.buffer = audioBuffer;
        source.connect(audioCtx.destination);
        source.start();
      };

      ws.onerror = (err) => {
        document.getElementById('status').textContent = '❌ 連線錯誤';
        document.getElementById('status').className = 'status error';
        console.error('WebSocket error:', err);
      };

      ws.onclose = () => {
        document.getElementById('status').textContent = '⚪ 連線已中斷';
        document.getElementById('status').className = 'status';
        document.getElementById('btnStart').style.display = 'inline-block';
        document.getElementById('btnStop').style.display = 'none';
        isPlaying = false;
      };
    }

    function stopAudio() {
      isPlaying = false;
      if (ws) {
        ws.close();
        ws = null;
      }
      if (audioCtx) {
        audioCtx.close();
        audioCtx = null;
      }
      document.getElementById('status').textContent = '點擊按鈕開始收聽';
      document.getElementById('status').className = 'status';
      document.getElementById('btnStart').style.display = 'inline-block';
      document.getElementById('btnStop').style.display = 'none';
      document.getElementById('volumeFill').style.width = '0%';
    }
  </script>
</body>
</html>
)rawliteral";

// ========== I2S 初始化 ==========
void i2sInit() {
  // I2S 設定
  const i2s_config_t i2s_config = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),  // 主模式 + 接收
    .sample_rate = SAMPLE_RATE,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,          // INMP441 輸出 32-bit
    .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,           // 只用左聲道（L/R 接 GND）
    .communication_format = I2S_COMM_FORMAT_STAND_I2S,     // 標準 I2S 格式
    .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,              // 中斷等級 1
    .dma_buf_count = 8,                                     // DMA 緩衝區數量
    .dma_buf_len = BUFFER_LEN,                              // 每個緩衝區的樣本數
    .use_apll = false,                                      // 不使用 APLL
    .tx_desc_auto_clear = false,
    .fixed_mclk = 0
  };

  // I2S 腳位設定
  const i2s_pin_config_t pin_config = {
    .bck_io_num = I2S_SCK,          // BCLK
    .ws_io_num = I2S_WS,            // LRCLK
    .data_out_num = I2S_PIN_NO_CHANGE,  // 不用輸出（沒有喇叭）
    .data_in_num = I2S_SD            // 資料輸入
  };

  // 安裝 I2S 驅動
  esp_err_t err = i2s_driver_install(I2S_PORT, &i2s_config, 0, NULL);
  if (err != ESP_OK) {
    Serial.printf("I2S 驅動安裝失敗: %d\n", err);
    return;
  }

  // 設定 I2S 腳位
  err = i2s_set_pin(I2S_PORT, &pin_config);
  if (err != ESP_OK) {
    Serial.printf("I2S 腳位設定失敗: %d\n", err);
    return;
  }

  Serial.println("I2S 初始化完成");
}

// ========== WebSocket 事件處理 ==========
void webSocketEvent(uint8_t num, WStype_t type, uint8_t* payload, size_t length) {
  switch (type) {
    case WStype_DISCONNECTED:
      Serial.printf("[%u] 已斷線\n", num);
      clientConnected = false;
      break;
    case WStype_CONNECTED:
      Serial.printf("[%u] 已連線\n", num);
      clientConnected = true;
      break;
    default:
      break;
  }
}

// ========== 讀取 I2S 並轉換為 16-bit ==========
void readAndSendAudio() {
  if (!clientConnected) return;

  size_t bytesRead = 0;

  // 從 I2S 讀取 32-bit 原始資料
  esp_err_t err = i2s_read(I2S_PORT, rawSamples, sizeof(rawSamples), &bytesRead, portMAX_DELAY);
  if (err != ESP_OK) return;

  int samplesRead = bytesRead / sizeof(int32_t);

  // 將 32-bit 轉換成 16-bit
  // INMP441 的有效資料在高位元，右移 16 位取出
  for (int i = 0; i < samplesRead; i++) {
    samples16[i] = (int16_t)(rawSamples[i] >> 16);
  }

  // 透過 WebSocket 發送 16-bit 音訊資料
  webSocket.broadcastBIN((uint8_t*)samples16, samplesRead * sizeof(int16_t));
}

// ========== setup() ==========
void setup() {
  Serial.begin(115200);
  Serial.println("\n=== ESP32 INMP441 音訊串流 ===");

  // 連接 WiFi
  WiFi.begin(ssid, password);
  Serial.print("正在連接 WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println();
  Serial.print("WiFi 已連線，IP 位址：");
  Serial.println(WiFi.localIP());

  // 初始化 I2S
  i2sInit();

  // 設定 HTTP 伺服器
  server.on("/", []() {
    server.send_P(200, "text/html", indexHtml);
  });
  server.begin();
  Serial.println("HTTP 伺服器已啟動（埠 80）");

  // 設定 WebSocket 伺服器
  webSocket.begin();
  webSocket.onEvent(webSocketEvent);
  Serial.println("WebSocket 伺服器已啟動（埠 81）");

  Serial.println("\n請用瀏覽器開啟以下網址：");
  Serial.print("http://");
  Serial.println(WiFi.localIP());
}

// ========== loop() ==========
void loop() {
  server.handleClient();
  webSocket.loop();
  readAndSendAudio();
}