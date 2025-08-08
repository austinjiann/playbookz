# Sports Highlight Generator Web App

AI-powered sports highlight generation with a user-friendly web interface. Upload any football/soccer video and get automatically generated highlights using advanced audio analysis and Gemini Vision API validation.

## Features

- 🎬 **Universal Video Support**: Works with any football video (phone recordings, broadcast footage, amateur videos)
- 🎵 **Audio-Based Detection**: Analyzes crowd noise and audio peaks to identify exciting moments
- 🤖 **AI Validation**: Uses Gemini Vision API to validate and classify detected moments
- 🌐 **Web Interface**: Simple drag-and-drop upload with real-time progress tracking
- 📊 **Live Monitoring**: Real-time logs and progress via Server-Sent Events
- 🎛️ **Configurable**: Adjustable sensitivity, clip length, and AI thresholds
- 🐳 **Docker Ready**: Complete containerized deployment with Nginx

## Quick Start

### 1. Prerequisites

- Python 3.11+
- FFmpeg (for video processing)
- [Gemini API Key](https://aistudio.google.com/app/apikey) from Google AI Studio

### 2. Installation

```bash
# Clone repository
git clone <repository-url>
cd football-highlights

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

### 3. Configuration

Create `.env` file with:

```env
GEMINI_API_KEY=your_gemini_api_key_here
MAX_UPLOAD_MB=2048
MAX_CONCURRENT_JOBS=2
JOB_TIMEOUT_HOURS=2
STRUCTURED_PROGRESS=true
```

### 4. Run the Application

```bash
# Development mode
uvicorn web.main:app --reload --host 0.0.0.0 --port 8000

# Production mode
uvicorn web.main:app --host 0.0.0.0 --port 8000
```

Visit http://localhost:8000 to access the web interface.

## Docker Deployment

### Quick Start with Docker

```bash
# Set environment variable
export GEMINI_API_KEY="your_api_key_here"

# Build and run
docker-compose up -d

# View logs
docker-compose logs -f
```

### Production Deployment

1. **Configure environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your settings
   ```

2. **Deploy with Docker Compose**:
   ```bash
   docker-compose up -d
   ```

3. **Setup reverse proxy** (optional):
   - The included `nginx.conf` provides production-ready configuration
   - Handles large file uploads (2GB+)
   - Optimizes SSE connections and static file serving

## API Endpoints

### Upload Video
```bash
POST /api/upload
Content-Type: multipart/form-data

# Form fields:
# - file: video file
# - debug: boolean (optional)
# - pre_sec: number (optional)
# - post_sec: number (optional) 
# - merge_window_sec: number (optional)
# - gemini_pos_threshold: number (optional)
```

### Job Status
```bash
GET /api/jobs/{job_id}/status
```

### Progress Stream (SSE)
```bash
GET /api/jobs/{job_id}/progress/stream
```

### Download Result
```bash
GET /api/jobs/{job_id}/result
```

### Health Check
```bash
GET /api/health
```

## Configuration Options

### Web UI Settings

- **Pre-clip seconds**: Time before exciting moment to include (default: 12s)
- **Post-clip seconds**: Time after exciting moment to include (default: 12s)
- **Merge window**: Maximum gap between moments to merge into one clip (default: 4s)
- **AI threshold**: Sensitivity for Gemini excitement detection (default: 0.4)
- **Debug mode**: Generates HTML preview of detected moments

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GEMINI_API_KEY` | - | **Required** Google Gemini API key |
| `MAX_UPLOAD_MB` | 2048 | Maximum upload file size in MB |
| `MAX_CONCURRENT_JOBS` | 2 | Maximum simultaneous processing jobs |
| `JOB_TIMEOUT_HOURS` | 2 | Job processing timeout |
| `STRUCTURED_PROGRESS` | true | Enable structured progress output |
| `ADMIN_KEY` | admin123 | Key for admin operations |

## How It Works

### Processing Pipeline

1. **Upload & Validation**: File uploaded with streaming to handle large files
2. **Audio Analysis**: Extract audio and detect excitement peaks using RMS envelope and bandpass filtering
3. **AI Validation**: Send video frames at peak timestamps to Gemini Vision API for validation
4. **Clip Generation**: Create highlight clips around validated moments with smart merging
5. **Video Rendering**: Concatenate clips into final highlights video using MoviePy

### Architecture

- **FastAPI**: Modern async web framework with automatic OpenAPI docs
- **SQLModel**: Type-safe database operations with SQLite
- **ThreadPoolExecutor**: Background job processing with proper resource management
- **Server-Sent Events**: Real-time progress updates and log streaming
- **HTMX**: Dynamic frontend updates without complex JavaScript
- **Docker**: Containerized deployment with Nginx reverse proxy

## Development

### Project Structure

```
├── web/
│   ├── main.py           # FastAPI application
│   ├── models.py         # Database models
│   ├── workers.py        # Job processing
│   ├── storage.py        # File handling
│   ├── progress.py       # SSE streaming
│   └── templates/
├── data/jobs/            # Job storage
├── tools/                # CLI utilities
├── tests/                # Test suite
└── config.py             # Configuration
```

### Testing

```bash
# Run test suite
pytest tests/ -v --asyncio-mode=auto

# Test specific functionality
pytest tests/test_audio_pipeline.py::TestIntegration -v

# Test CLI directly
python gen_highlights.py --video test.mp4 --mode audio --debug
```

### Debugging

1. **Check logs**: `/api/jobs/{job_id}/logs/stream`
2. **Enable debug mode**: Generates HTML preview with frame analysis
3. **Use CLI tools**: Direct testing with `python audio_analysis.py video.mp4`
4. **Health check**: `/api/health` shows system status

## Troubleshooting

### Common Issues

1. **"GEMINI_API_KEY required"**
   - Get API key from [Google AI Studio](https://aistudio.google.com/app/apikey)
   - Set in `.env` file or environment variable

2. **Large file upload fails**
   - Check `MAX_UPLOAD_MB` setting
   - Ensure sufficient disk space in `data/` directory
   - For Docker: verify volume mounts

3. **FFmpeg not found**
   - Install FFmpeg system-wide
   - For Docker: included in container image
   - Set `IMAGEIO_FFMPEG_EXE` environment variable if needed

4. **Jobs stuck in processing**
   - Check job logs via SSE stream
   - Restart application (processing jobs auto-marked as failed)
   - Verify Gemini API key and quota

5. **Memory issues with large videos**
   - Reduce `MAX_CONCURRENT_JOBS`
   - Increase system memory allocation
   - Videos are processed in chunks to minimize memory usage

### Production Considerations

- **Monitoring**: Use `/api/health` endpoint for health checks
- **Backup**: Job database at `data/jobs.db`, results in `data/jobs/`
- **Cleanup**: Use `/admin/cleanup` to remove old jobs
- **Scaling**: Horizontal scaling requires shared storage for job data
- **Security**: Change `ADMIN_KEY`, use HTTPS in production

## Tiny xG (StatsBomb Open Data)

This repository includes a lightweight Expected Goals (xG) model trained on StatsBomb's open data.

### Setup

Install dependencies:
```bash
pip install -r requirements.txt
```

Run training once to create coefficients:
```bash
python scripts/train_xg.py
```

### Usage

```python
from models.xg_model import predict_xg
xg = predict_xg(x=102.3, y=37.8, body_part="left_foot", is_set_piece=False, is_open_play=True)
print(f"xG: {xg:.3f}")
```

### Notes

- **No dataset needed at runtime** - coefficients are baked into `data/xg_coeffs.json`
- **StatsBomb pitch coordinates**: 120x80 field, goalposts at y={36,44}
- **First run may take a few minutes** to download and process open data
- **If rate-limited**: Re-run script, it will automatically use a smaller competition subset

## xG Integration in Highlight Generation

The xG model can be integrated with the main highlight detection pipeline to enhance shot detection and ranking.

### Usage

Enable xG integration when generating highlights:

```bash
# Basic xG integration (30% weight)
python gen_highlights.py --video match.mp4 --mode audio --use-xg

# Custom xG weighting (50% weight)
python gen_highlights.py --video match.mp4 --mode audio --use-xg --xg-weight 0.5

# Debug mode to see xG calculations
python gen_highlights.py --video match.mp4 --mode audio --use-xg --debug
```

### How It Works

1. **Shot Detection**: Gemini Vision API identifies exciting moments in frames
2. **Coordinate Estimation**: Placeholder mapping estimates shot location on pitch
3. **xG Calculation**: Uses trained model to compute goal probability
4. **Enhanced Ranking**: Blends Gemini confidence with xG scores using configurable weight

### Configuration Options

| Flag | Default | Description |
|------|---------|-------------|
| `--use-xg` | False | Enable xG integration for highlight ranking |
| `--xg-weight` | 0.3 | Weight (0-1) for blending xG with Gemini confidence |

### Ranking Formula

```python
final_score = (1 - xg_weight) * gemini_confidence + xg_weight * xg_value
```

**Example**: With `--xg-weight 0.3`:
- Gemini confidence: 0.8 (80%)
- xG value: 0.2 (20% goal probability)  
- Final score: 0.7 × 0.8 + 0.3 × 0.2 = 0.62

### Debug Output

When using `--debug --use-xg`, you'll see detailed logging:

```
Shot at 45.2s: xG=0.234 (coords: 108.0, 40.0)
Clip 45.0s: Gemini=0.800 + xG=0.234 → Blended=0.630
```

### JSON Output

xG data appears in run summaries:

```json
{
  "clips": [{
    "start_time": 45.0,
    "confidence": 0.8,
    "xg": 0.234,
    "blended_score": 0.630,
    "coordinates": [108.0, 40.0]
  }]
}
```

### Coordinate Mapping

Currently uses placeholder coordinates (penalty area: x=108, y=40) for immediate testing. Future enhancements will include:

- Homography-based pitch mapping
- Computer vision-based coordinate detection  
- Context-aware position estimation

### Performance

- **Backward Compatible**: Default behavior unchanged when `--use-xg` not used
- **Graceful Fallback**: Continues without xG if model unavailable
- **Efficient**: xG calculated only for exciting frames identified by Gemini

## License

This project is licensed under the MIT License.

## Support

For issues and questions:
1. Check existing GitHub issues
2. Review CLAUDE.md for development guidance
3. Enable debug mode for detailed analysis
4. Use health check endpoint for system diagnostics