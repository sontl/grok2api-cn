# Grok2API

A Grok2API reconstructed based on **FastAPI**, fully compatible with the latest web call formats, supporting streaming conversations, image generation, image editing, web search, deep thinking, account pool concurrency, and automatic load balancing integrated.

<br>

## Usage Instructions

### Call Counts and Quotas

- **Basic Account**: Free usage of **80 times / 20 hours**
- **Super Account**: Quota to be determined (author hasn't tested)
- The system automatically balances call counts across accounts. Usage and status can be viewed in real-time on the **Management Page**

### Image Generation Functionality

- Enter content like "draw me a moon" in the conversation to automatically trigger image generation
- Each time returns **two images in Markdown format**, consuming 4 quota units in total
- **Note: Grok's image direct links are restricted by 403, so the system automatically caches images locally. You must set `Base Url` correctly to ensure images display normally!**

### Video Generation Functionality
- Select the `grok-imagine-0.9` model and pass in an image and prompt (format is consistent with OpenAI's image analysis call format)
- Return format is `<video src="{full_video_url}" controls="controls"></video>`
- **Note: Grok's video direct links are restricted by 403, so the system automatically caches videos locally. You must set `Base Url` correctly to ensure videos display normally!**

```
curl https://your-server-address/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $GROK2API_API_KEY" \
  -d '{
    "model": "grok-imagine-0.9",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "Make the sun rise"
          },
          {
            "type": "image_url",
            "image_url": {
              "url": "https://your-image.jpg"
            }
          }
        ]
      }
    ]
  }'
```

### About `x_statsig_id`

- `x_statsig_id` is a token used by Grok for anti-robot verification, with reverse engineering resources available for reference
- **Beginners are advised not to modify the configuration, just keep the default values**
- Attempted to bypass 403 automatically with Camoufox to obtain the ID, but Grok now restricts `x_statsig_id` for non-logged in users, so it's deprecated and fixed values are used to be compatible with all requests

<br>

## How to Deploy

### docker-compose

```yaml
services:
  grok2api:
    image: ghcr.io/chenyme/grok2api:latest
    ports:
      - "8000:8000"
    volumes:
      - grok_data:/app/data
      - ./logs:/app/logs
    environment:
      # =====Storage Mode: file, mysql, or redis=====
      - STORAGE_MODE=file
      # =====Database Connection URL (Required only when STORAGE_MODE=mysql or redis)=====
      # - DATABASE_URL=mysql://user:password@host:3306/grok2api

      ## MySQL format: mysql://user:password@host:port/database
      ## Redis format: redis://host:port/db or redis://user:password@host:port/db

volumes:
  grok_data:
```

### Environment Variable Description

| Environment Variable | Required | Description                                    | Example |
|----------------------|----------|-----------------------------------------------|---------|
| STORAGE_MODE         | No       | Storage mode: file/mysql/redis                | file    |
| DATABASE_URL         | No       | Database connection URL (Required for MySQL/Redis mode) | mysql://user:pass@host:3306/db |

**Storage Modes:**
- `file`: Local file storage (default)
- `mysql`: MySQL database storage, requires DATABASE_URL
- `redis`: Redis cache storage, requires DATABASE_URL

<br>

## API Interface Description

> Fully compatible with OpenAI official API interface, API requests require authentication via **Authorization header**

| Method | Endpoint                      | Description                         | Authentication Required |
|--------|-------------------------------|-------------------------------------|------------------------|
| POST   | `/v1/chat/completions`        | Create chat completion (streaming/non-streaming) | ✅ |
| GET    | `/v1/models`                  | Get all supported models            | ✅ |
| GET    | `/images/{img_path}`          | Get generated image file            | ❌ |

<br>

<details>
<summary>Management and Statistics Interfaces (Expand for more) </summary>

| Method | Endpoint             | Description               | Authentication |
|--------|----------------------|---------------------------|----------------|
| GET    | /login               | Administrator login page  | ❌             |
| GET    | /manage              | Management console page   | ❌             |
| POST   | /api/login           | Administrator login authentication | ❌       |
| POST   | /api/logout          | Administrator logout      | ✅             |
| GET    | /api/tokens          | Get token list            | ✅             |
| POST   | /api/tokens/add      | Batch add tokens          | ✅             |
| POST   | /api/tokens/delete   | Batch delete tokens       | ✅             |
| GET    | /api/settings        | Get system configuration| ✅             |
| POST   | /api/settings        | Update system configuration | ✅         |
| GET    | /api/cache/size      | Get cache size            | ✅             |
| POST   | /api/cache/clear     | Clear all cache           | ✅             |
| POST   | /api/cache/clear/images | Clear image cache      | ✅             |
| POST   | /api/cache/clear/videos | Clear video cache       | ✅             |
| GET    | /api/stats           | Get statistics            | ✅             |
| POST   | /api/tokens/tags     | Update token tags         | ✅             |
| POST   | /api/tokens/note     | Update token notes        | ✅             |
| POST   | /api/tokens/test     | Test token availability   | ✅             |
| GET    | /api/tokens/tags/all | Get all tag list          | ✅             |
| GET    | /api/storage/mode    | Get storage mode info     | ✅             |

</details>

<br>

## Available Models Overview

| Model Name              | Count | Account Type | Image Generation/Edit | Deep Thinking | Web Search | Video Generation |
|-------------------------|-------|--------------|-----------------------|---------------|------------|------------------|
| `grok-4.1`              | 1     | Basic/Super  | ✅                    | ✅            | ✅         | ❌               |
| `grok-4.1-thinking`     | 1     | Basic/Super  | ✅                    | ✅            | ✅         | ❌               |
| `grok-imagine-0.9`      | -     | Basic/Super  | ✅                    | ❌            | ❌         | ✅               |
| `grok-4-fast`           | 1     | Basic/Super  | ✅                    | ✅            | ✅         | ❌               |
| `grok-4-fast-expert`    | 4     | Basic/Super  | ✅                    | ✅            | ✅         | ❌               |
| `grok-4-expert`         | 4     | Basic/Super  | ✅                    | ✅            | ✅         | ❌               |
| `grok-4-heavy`          | 1     | Super        | ✅                    | ✅            | ✅         | ❌               |
| `grok-3-fast`           | 1     | Basic/Super  | ✅                    | ❌            | ✅         | ❌               |

<br>

## Configuration Parameter Description

> After service starts, log in to `/login` management backend for parameter configuration

| Parameter Name            | Scope  | Required | Description                                    | Default Value |
|---------------------------|--------|----------|-----------------------------------------------|---------------|
| admin_username            | global | No       | Management backend login username             | "admin"       |
| admin_password            | global | No       | Management backend login password             | "admin"       |
| log_level                 | global | No       | Log level: DEBUG/INFO/...                     | "INFO"        |
| image_mode                | global | No       | Image return mode: url/base64                 | "url"         |
| image_cache_max_size_mb   | global | No       | Maximum image cache capacity (MB)             | 512           |
| video_cache_max_size_mb   | global | No       | Maximum video cache capacity (MB)             | 1024          |
| base_url                  | global | No       | Service base URL/image access base            | ""            |
| api_key                   | grok   | No       | API key (optional for enhanced security)      | ""            |
| proxy_url                 | grok   | No       | HTTP proxy server address                     | ""            |
| stream_chunk_timeout      | grok   | No       | Streaming chunk timeout (seconds)             | 120           |
| stream_first_response_timeout | grok | No     | Streaming first response timeout (seconds)    | 30            |
| stream_total_timeout      | grok   | No       | Streaming total timeout (seconds)             | 600           |
| cf_clearance              | grok   | No       | Cloudflare security token                     | ""            |
| x_statsig_id              | grok   | Yes      | Anti-robot unique identifier                  | "ZTpUeXBlRXJyb3I6IENhbm5vdCByZWFkIHByb3BlcnRpZXMgb2YgdW5kZWZpbmVkIChyZWFkaW5nICdjaGlsZE5vZGVzJyk=" |
| filtered_tags             | grok   | No       | Filter response tags (comma separated)        | "xaiartifact,xai:tool_usage_card,grok:render" |
| show_thinking             | grok   | No       | Show thinking process true(show)/false(hide)  | true          |
| temporary                 | grok   | No       | Session mode true(temporary)/false            | true          |

<br>

## ⚠️ Notes

This project is for learning and research purposes only. Please comply with the relevant terms of use!

<br>

> This project is restructured based on the following projects for learning purposes, special thanks to: [LINUX DO](https://linux.do), [VeroFess/grok2api](https://github.com/VeroFess/grok2api), [xLmiler/grok2api_python](https://github.com/xLmiler/grok2api_python)