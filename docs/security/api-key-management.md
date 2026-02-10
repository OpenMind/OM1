# docs/security/api-key-management.md

# Secure API Key Management

## ⚠️ Security Best Practices

**NEVER commit API keys to Git!**

OM1 supports multiple secure methods for API key storage:

## Methods (in priority order)

### 1. Environment Variables (Recommended)
```bash
export OPENMIND_API_KEY="om-your-key-here"
export OPENAI_API_KEY="sk-your-key-here"

# Or add to ~/.bashrc for persistence
echo 'export OPENMIND_API_KEY="om-..."' >> ~/.bashrc
```

### 2. OS Keychain (Most Secure)
```bash
# Set API key
python -m src.cli.api_keys set openmind

# Get API key (masked)
python -m src.cli.api_keys get openmind

# List all configured keys
python -m src.cli.api_keys list
```

### 3. .env File (Fallback)
```bash
# Create .env file
cp env.example .env

# Edit and add keys
nano .env
```

**Important:** `.env` file has 600 permissions (owner read/write only)

## Migration Guide

### From config files to secure storage:
```bash
# 1. Backup your config
cp config/spot.json5 config/spot.json5.backup

# 2. Set API key securely
python -m src.cli.api_keys set openmind

# 3. Remove from config file
# Edit config/spot.json5 and replace:
"api_key": "om-abc123..."
# With:
"api_key": "openmind_free"  # Placeholder

# 4. Test
uv run src/run.py conversation
```

## Security Features

- ✅ OS-level keychain integration
- ✅ API key format validation
- ✅ Automatic placeholder detection
- ✅ Secure file permissions
- ✅ Key masking in logs
- ✅ Multiple fallback sources

## Troubleshooting

**Error: "API key required for openmind"**
- Set via: `python -m src.cli.api_keys set openmind`
- Or export: `export OPENMIND_API_KEY="om-..."`

**Error: "Invalid API key format"**
- Check key starts with correct prefix (om-, sk-, etc.)
- Ensure no extra spaces or quotes
