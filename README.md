# Video Censor / 视频审核打码服务

[English](#english) | [中文](#中文)

---

## English

AI-powered video content moderation and auto-censoring service for short-form dramas. Automatically detects nudity, violent content, prohibited text in subtitles, and applies mosaic/overlay censoring.

### Features

- **Multi-layer Detection**: NudeNet (nudity) + Volcano VLM (content audit) + RapidOCR (subtitle text)
- **Smart Censoring**: Pixelated mosaic for body regions, solid color overlay for subtitle text
- **VLM Cross-validation**: Reduces false positives by confirming borderline detections with VLM
- **GPU Accelerated**: NVIDIA GPU support via onnxruntime-gpu for NudeNet and OCR
- **Highlight Detection**: Identifies key moments for video editing
- **Async Processing**: Celery workers with concurrent VLM API calls
- **Web UI**: Upload, monitor progress, preview and download results

### Architecture

```
┌─────────┐     ┌─────────┐     ┌───────┐
│ Frontend │────▶│ Backend │────▶│ Redis │
│ (Nginx)  │     │(FastAPI)│     │       │
└─────────┘     └─────────┘     └───┬───┘
                                     │
                              ┌──────▼──────┐
                              │   Worker    │
                              │  (Celery)   │
                              ├─────────────┤
                              │ NudeNet GPU │
                              │ RapidOCR GPU│
                              │ Volcano VLM │
                              └─────────────┘
```

### Requirements

- Docker & Docker Compose
- NVIDIA GPU (RTX 3060+ recommended) with drivers installed
- NVIDIA Container Toolkit (`nvidia-docker`)
- Volcano Engine API Key (for VLM content audit)

### Quick Start

```bash
# Clone
git clone https://github.com/alex1678747287/video-censor.git
cd video-censor

# Configure
cp .env.example .env
# Edit .env and set your VOLCANO_API_KEY

# Launch
docker compose up -d

# Access
open http://localhost
```

### API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/tasks` | Upload video file |
| GET | `/api/tasks` | List tasks (paginated) |
| GET | `/api/tasks/{id}` | Get task detail + violations |
| GET | `/api/tasks/{id}/download` | Download censored video |
| DELETE | `/api/tasks/{id}` | Delete task and files |

### Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `VOLCANO_API_KEY` | - | Volcano Engine API key |
| `VOLCANO_MODEL` | `doubao-seed-2-0-lite-260215` | VLM model ID |
| `REDIS_URL` | `redis://redis:6379/0` | Redis connection |

### Tech Stack

- **Backend**: Python 3.12, FastAPI, Celery, SQLAlchemy
- **Detection**: NudeNet 3.4, RapidOCR, Volcano VLM (doubao-seed)
- **Processing**: OpenCV, FFmpeg
- **Frontend**: Vue 3, Element Plus (CDN)
- **Infra**: Docker, Nginx, Redis, NVIDIA GPU

---

## 中文

基于 AI 的微短剧视频内容审核与自动打码服务。自动检测裸露、暴力内容、字幕违规文字，并进行马赛克/遮挡处理。

### 功能特性

- **多层检测**: NudeNet（裸露检测）+ 火山 VLM（内容审核）+ RapidOCR（字幕文字）
- **智能打码**: 身体区域像素化马赛克，字幕文字纯色遮挡
- **VLM 交叉验证**: 边界检测通过 VLM 二次确认，减少误打码
- **GPU 加速**: 通过 onnxruntime-gpu 支持 NVIDIA GPU 加速
- **高光检测**: 自动识别视频精彩片段
- **异步处理**: Celery 工作进程 + VLM 并发调用
- **Web 界面**: 上传、进度监控、预览和下载

### 环境要求

- Docker & Docker Compose
- NVIDIA 显卡（推荐 RTX 3060+）及驱动
- NVIDIA Container Toolkit
- 火山引擎 API Key（用于 VLM 内容审核）

### 快速开始

```bash
# 克隆
git clone https://github.com/alex1678747287/video-censor.git
cd video-censor

# 配置
cp .env.example .env
# 编辑 .env 设置 VOLCANO_API_KEY

# 启动
docker compose up -d

# 访问
open http://localhost
```

### API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/tasks` | 上传视频文件 |
| GET | `/api/tasks` | 任务列表（分页）|
| GET | `/api/tasks/{id}` | 任务详情 + 违规信息 |
| GET | `/api/tasks/{id}/download` | 下载打码后视频 |
| DELETE | `/api/tasks/{id}` | 删除任务及文件 |

### 检测流程

1. 提取视频帧（2fps + 场景切换检测）
2. NudeNet + OCR 并行检测
3. VLM 确认边界检测 + 字幕语义审核
4. VLM 画面审核 + 高光检测（并行）
5. OpenCV 逐帧打码 + FFmpeg 重编码

### 成本参考

60 秒视频约 ¥0.1-0.15（火山 VLM API 费用），预算上限可控制在 ¥0.3 以内。

### License

MIT
