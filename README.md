# Video Commentary Bot

A Telegram bot that generates engaging commentary for videos using AI.

## Features

- Multiple commentary styles (Documentary, Energetic, Analytical, Storyteller)
- Intelligent video analysis using Google Cloud Vision AI
- Natural language commentary generation with GPT-4
- Professional audio synthesis with Google Text-to-Speech
- Automatic video processing and generation

## Prerequisites

- Python 3.10 or higher
- FFmpeg
- Google Cloud account with Vision and Text-to-Speech APIs enabled
- OpenAI API key
- Telegram Bot token
- Cloudinary account

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/MrEmanAhmad/video_processing_bot_Pk_ur.git
   cd video_processing_bot_Pk_ur
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   - Copy `.env.example` to `.env`
   - Fill in your API keys and credentials

## Configuration

### Required Environment Variables

- `TELEGRAM_BOT_TOKEN`: Your Telegram bot token from BotFather
- `OPENAI_API_KEY`: OpenAI API key for commentary generation
- `DEEPSEEK_API_KEY`: DeepSeek API key (optional alternative to OpenAI)
- `GOOGLE_APPLICATION_CREDENTIALS_JSON`: Google Cloud credentials JSON string
- `CLOUDINARY_CLOUD_NAME`: Cloudinary cloud name
- `CLOUDINARY_API_KEY`: Cloudinary API key
- `CLOUDINARY_API_SECRET`: Cloudinary API secret

### Google Cloud Setup

1. Create a Google Cloud Project and enable:
   - Cloud Vision API
   - Cloud Text-to-Speech API

2. Create a service account with necessary permissions:
   - Cloud Vision API User
   - Cloud Text-to-Speech API User

3. Download the service account key JSON and format it as a single line for the `GOOGLE_APPLICATION_CREDENTIALS_JSON` environment variable.

## Usage

1. Start the bot:
   ```bash
   python bot.py
   ```

2. In Telegram:
   - Start a chat with your bot
   - Use /start to select commentary style
   - Send a video or video URL
   - Wait for the bot to process and return the video with commentary

## Docker Support

Build and run with Docker:

```bash
# Build
docker build -t video-commentary-bot .

# Run
docker run -d --env-file .env video-commentary-bot
```

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License - See LICENSE file for details

## Support

For support, please open an issue in the GitHub repository. 