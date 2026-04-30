# ESP32 ICS43434 双任务录音程序详细流程说明

## 📋 程序概览

这是一个**两任务+队列**的设计，采用**生产者-消费者**模式：
- **capture_task** (生产者)：从I2S麦克风读取音频数据
- **writer_task** (消费者)：将数据写入SD卡

两个任务通过 **FreeRTOS Queue** 异步通信，避免数据丢失和加速播放问题。

---

## 🏗️ 内存结构

### 1. 固定内存池（预分配，避免运行时malloc失败）

```c
// 16个音频块
static audio_block_t g_audio_blocks[AUDIO_BLOCK_COUNT];
//  每个块结构
typedef struct {
    size_t sample_count;                    // 当前块内样本数 (0-1024)
    int16_t samples[AUDIO_BLOCK_SAMPLES];   // 1024个PCM样本
} audio_block_t;

// I2S读取缓冲 (8KB)
static int32_t g_i2s_read_buffer[I2S_READ_BYTES / sizeof(int32_t)];

// SD卡文件IO缓冲 (32KB)
static char g_file_io_buffer[FILE_IO_BUFFER_SIZE];
```

**内存分配**：
```
I2S读缓冲:     8 KB   (4096个int32_t)
16个块缓冲:    32 KB  (16 × 2048字节)
文件IO缓冲:    32 KB
────────────────────
总计:         ~73 KB  (远小于204KB可用DRAM)
```

### 2. 队列系统

```c
static QueueHandle_t q_free_blocks;    // 待填充的空块
static QueueHandle_t q_full_blocks;    // 待写入的满块
```

**队列容量**：
- `q_free_blocks`：最多16个项 (初始全满)
- `q_full_blocks`：最多17个项 (包含sentinel)

---

## 🔄 完整运行流程

### 第0步：初始化（主任务）

```
app_main()
    ↓
[初始化流程]
1. dbg_pin_init()          → 初始化GPIO12(capture指示)、GPIO18(writer指示)
2. sd_card_init()          → 挂载SD卡到/sdcard
3. i2s_init_equivalent_to_ino()  → 初始化I2S DMA + 时钟
4. audio_pipeline_init()   → 创建队列，初始化块池
           ↓
[块池初始状态]
q_free_blocks:  ████████████████ (16个空块)
q_full_blocks:  ░░░░░░░░░░░░░░░░░ (0个)
```

### 第1步：启动两个任务

```c
// 从任务计划表创建任务
for task in g_task_plan:
    xTaskCreatePinnedToCore(
        capture_task,           // 核心1，优先级10（高）
        writer_task             // 核心0，优先级6（中）
    )
```

---

## 🎤 Capture Task（核心1，优先级10）

**目标**：从I2S持续读取音频数据，填充块，发送到队列

### 时间轴：

```
[TIME 0ms] 任务启动
────────────────────────────────────────
GPIO12 = 1  (指示录音中)

[TIME 0-22ms] 第1轮循环
────────────────────────────────────────
1. xQueueReceive(q_free_blocks, &blk)  
   → 获得空块 (blk[0])
   
2. 从I2S读取数据 (8KB = 2048个int32_t)：
   i2s_channel_read()
   ├─ 从DMA读取8192字节
   └─ 存入g_i2s_read_buffer[]
   
3. 数据转换：32bit → 16bit
   for i in 0..2047:
       raw = g_i2s_read_buffer[i]
       pcm = (int16_t)(raw >> 16)  // 提取高16bit
       blk->samples[blk->sample_count++] = pcm
   
4. 块满足条件(sample_count >= 1024)？
   YES:  xQueueSend(q_full_blocks, &blk)  // 发送给writer
         xQueueReceive(q_free_blocks, &blk) // 获取新空块
   NO:   继续在同一块中累积

示例：
需要 88,200 样本（2秒 @ 44.1kHz）
一次读：2048个样本 (I2S_READ_BYTES / 4)
一块容量：1024个样本
轮数：88,200 / 2048 ≈ 43次循环
────────────────────────────────────────

[TIME 20-40ms] 第2轮循环
────────────────────────────────────────
块[0] 状态：
  ├─ 前1024个样本 → 已发送到q_full_blocks
  └─ 后1024个样本 → 正在填充

块[1] 状态：在q_free_blocks中等待

块[2] 状态：在writer_task中写入SD卡
────────────────────────────────────────

... (持续43次循环) ...

[TIME 2000ms] 录音完成
────────────────────────────────────────
total_samples_captured = 88,200
最后一个不完整的块发送到q_full_blocks
发送sentinel (NULL)表示结束
GPIO12 = 0
capture_task 删除自己
```

### Capture Task伪代码

```c
void capture_task(void *arg) {
    target_samples = 88200
    total_samples_captured = 0
    
    xQueueReceive(q_free_blocks, &blk)  // 获取空块
    blk->sample_count = 0
    
    while (total_samples_captured < target_samples) {
        // 1. 读I2S
        i2s_channel_read(g_i2s_read_buffer, 8192, &bytes_read)
        
        // 2. 转换数据
        n = bytes_read / 4  // int32_t个数
        for (i = 0; i < n; i++) {
            pcm = (int16_t)(g_i2s_read_buffer[i] >> 16)
            blk->samples[blk->sample_count++] = pcm
            total_samples_captured++
            
            // 3. 块满了吗？
            if (blk->sample_count >= 1024) {
                xQueueSend(q_full_blocks, &blk)        // 放入待写队列
                xQueueReceive(q_free_blocks, &blk)     // 获取新块
                blk->sample_count = 0
            }
        }
    }
    
    // 4. 发送最后一块（可能不完整）
    if (blk->sample_count > 0) {
        xQueueSend(q_full_blocks, &blk)
    }
    
    // 5. 发送结束标志
    xQueueSend(q_full_blocks, NULL)  // sentinel
}
```

---

## 💾 Writer Task（核心0，优先级6）

**目标**：从队列接收数据块，写入SD卡WAV文件

### 时间轴：

```
[TIME 0ms+] 任务启动
────────────────────────────────────────
1. 打开SD卡文件
   wav_open_placeholder("/sdcard/audio.wav")
   └─ 写入44字节WAV头（暂时）
   
[TIME 0-1ms] 等待第一个块
────────────────────────────────────────
xQueueReceive(q_full_blocks, &blk, portMAX_DELAY)
(如果队列空，等待capture_task填充)

[TIME 1-5ms] 第1个块写入
────────────────────────────────────────
blk[0] 到达（包含1024个样本）
GPIO18 = 1  (指示写SD中)
fwrite(blk->samples, sizeof(int16_t), 1024, f)
GPIO18 = 0
g_total_samples_written += 1024

blk->sample_count = 0
xQueueSend(q_free_blocks, &blk)  // 放回空块池
────────────────────────────────────────

[TIME 5-10ms] 第2-140个块写入
────────────────────────────────────────
(继续接收、写入、回收块)
────────────────────────────────────────

[TIME 2000ms] 接收sentinel（结束标志）
────────────────────────────────────────
xQueueReceive() → blk = NULL
(循环退出)

[TIME 2000-2005ms] 更新WAV头
────────────────────────────────────────
wav_finalize(f, g_total_samples_written)
├─ 计算数据字节数：88200 × 2 = 176,400 bytes
├─ 填写WAV头结构
├─ fseek(f, 0)
├─ fwrite(hdr, 44)
└─ fclose(f)

最终文件大小：44 + 176,400 = 176,444 bytes
────────────────────────────────────────
```

### Writer Task伪代码

```c
void writer_task(void *arg) {
    path = "/sdcard/audio.wav"
    f = wav_open_placeholder(path)  // 写入临时头
    g_total_samples_written = 0
    
    while (1) {
        // 1. 获取待写块
        xQueueReceive(q_full_blocks, &blk, portMAX_DELAY)
        
        // 2. 检查是否为结束标志
        if (blk == NULL) {
            break
        }
        
        // 3. 写入SD卡
        GPIO18 = 1
        written = fwrite(blk->samples, 2, blk->sample_count, f)
        GPIO18 = 0
        
        g_total_samples_written += written
        
        // 4. 回收块到空块池
        blk->sample_count = 0
        xQueueSend(q_free_blocks, &blk)
    }
    
    // 5. 最终化WAV文件
    wav_finalize(f, g_total_samples_written)
    
    // 6. 通知主任务完成
    xTaskNotifyGive(g_main_task)
}
```

---

## 🔗 队列交互示意图

### 时刻 T1 (Capture快, Writer正在写)：

```
Capture Task                   Queue                    Writer Task
─────────────────────────────────────────────────────────────────
正在读I2S             q_free_blocks            正在fwrite块[5]
  ↓                    ├─ 块[2]
fill 块[1]            ├─ 块[3]
  ↓                    ├─ 块[4]
send 块[1]     →  send到q_full_blocks
                                              等待接收块[6]
                       q_full_blocks
                       ├─ 块[5] (正在写)
                       ├─ 块[6] (待写)
                       ├─ 块[7] (待写)
```

### 时刻 T2 (Capture等, Writer快)：

```
Capture Task                   Queue                    Writer Task
─────────────────────────────────────────────────────────────────
等待块         receive from q_free_blocks (阻塞)  写块[10]
                                                      ↓
                       q_free_blocks                 write完
                       ├─ 块[1] (just回收)           ↓
                       ├─ 块[2]                     send回
                       ├─ ...
                       └─ 块[16]
                                              等待下一个块
                       q_full_blocks
                       ├─ 块[11]
                       └─ 块[12]
```

---

## ⏱️ 时序关键点

### 为什么不会加速播放？

**关键**：`capture_task` 严格按照I2S实时时钟采样

```
I2S硬件生成数据速率：固定 44.1 kHz
└─ 不管software有多慢，I2S DMA会继续填充缓冲

capture_task 消费速率：
└─ 尽快读取DMA数据，转换成16bit
└─ 速率 ≥ I2S产生速率 (否则DMA溢出)

writer_task 消费速率：
└─ SD卡写入速度 (~100 KB/s)
└─ 比I2S产生速率慢，但有备缓冲
```

**分析**：
```
时间轴：
0ms      1ms      2ms      3ms      4ms      5ms      ...
│        │        │        │        │        │
I2S产生  44 samp  44 samp  44 samp  44 samp  44 samp
│
capture读 ~2000 samp/次 (8KB @ 44.1kHz ≈ 45ms的数据)
│
writer写  ~1024 samp/块 (延迟通过队列缓冲，但数据顺序保持)
```

块中的样本顺序 = 采样的实际顺序
不会出现乱序或跳跃

---

## 🎯 为什么采用双任务设计？

### 问题：边读边写（原方案）

```c
while (读取I2S) {
    转换数据
    写SD卡  ← 阻塞50ms  导致I2S DMA溢出！
}
```

**结果**：
- I2S DMA缓冲满 → 溢出
- 一些样本丢失 → 出现间隙或加速

### 解决：双任务+队列（新方案）

```
capture_task:        writer_task:
读I2S (1ms)         等待块 → 写入块 (50ms)
转换 (1ms)          不影响capture
发送队列 (0.1ms)
└─→ 总耗时 ~2ms      capture可继续
    < 一个块的     与writer并行
      传输时间
```

**优势**：
- ✅ capture任务干净快速
- ✅ writer不会堵塞capture
- ✅ 队列充当"缓冲垫"，吸收I/O延迟
- ✅ 双核心并行，充分利用处理器
- ✅ 数据顺序绝对保证

---

## 📊 数据统计

**2秒录音参数**：
```
采样率:        16000 Hz
时长:          10秒 (RECORD_TIME_SEC)
目标样本:      160,000
目标PCM数据:   320,000 bytes (312.5 KB)

块配置:
  单块容量:    1024样本 = 2048 bytes = 64ms数据
  块总数:      16块
  总缓冲:      32 KB (约0.8秒音频)

SD写入:
  一次fwrite:  1024样本 = 2048 bytes
  文件IO缓冲:  32 KB (16次write才触发磁盘)
```

---

## 🔍 调试方法

观察GPIO指示灯：

```
GPIO12 (capture中):    ████████████ (应持续10秒)
GPIO18 (writer中):     ▓▓▓▓░░▓▓▓░░░ (间断性发光)

如果GPIO12闪烁(不连续)：I2S溢出
如果GPIO18灭(不发光)：  SD卡太慢
```

---

## 总结流程

```
app_main()
    ↓
[初始化所有硬件和队列]
    ↓
[创建capture_task和writer_task]
    ↓
capture_task                          writer_task
  读I2S (持续10s)                       等待q_full_blocks
  ↓                                    ↓
  转换32→16bit                         写块到SD卡 (并发)
  ↓                               
  填充块→q_full_blocks ═════════→ 读块→转换→fwrite
  ↓                                  ↓
  回收块←q_free_blocks ←════════ 回收块
  ↓                                    ↓
  发送sentinel(NULL)              接收NULL→finalize WAV头
  ↓                                    ↓
  task删除                             task删除
                             ↓
              app_main继续→清理资源→返回
```

程序保证：**采样顺序 = 最终文件中的顺序** ✅
