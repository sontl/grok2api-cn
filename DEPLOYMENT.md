# Deployment Instructions

## Environment Setup

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` and fill in your actual values:
   ```bash
   # Storage mode: file, mysql, or redis
   STORAGE_MODE=mysql
   
   # Database connection URL (required when STORAGE_MODE=mysql or redis)
   DATABASE_URL=mysql://user:password@host:port/database
   ```

## Docker Deployment

1. Build and run with Docker Compose:
   ```bash
   docker-compose up -d
   ```

2. The application will be available at `http://localhost:8002`

## Environment Variables

- `STORAGE_MODE`: Storage backend (`file`, `mysql`, or `redis`)
- `DATABASE_URL`: Database connection string (required for mysql/redis modes)

### Database URL Formats

- **MySQL**: `mysql://user:password@host:port/database`
- **Redis**: `redis://host:port/db` or `redis://user:password@host:port/db`