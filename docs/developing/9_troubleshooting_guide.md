---
title: Troubleshooting Guide
description: "Guide to troubleshoot some common issues"
icon: wrench
---

### Common Issues & Quick Fixes

| Issue                   | Likely Cause                | Quick Fix                                                                 |
|-------------------------|-----------------------------|---------------------------------------------------------------------------|
| No Speech               | Permission issues           | Check the settings                                                        |
| No speech recognition   | Microphone not configured   | Check audio input settings                                                |
| Robot not moving        | Connection issue/Network issue            | Restart OM1/Robot and check your internet connection                                        |
| OpenSSL certificate issue | Security certificate not found | `uv pip install certifi` `export SSL_CERT_FILE=$(python3 -m certifi)` `export REQUESTS_CA_BUNDLE=$(python3 -m certifi)` |
| Error message during build: `fatal error: portaudio.h: No such file or directory compilation terminated. error: command '/usr/bin/gcc' failed with exit code 1` | The issue is due to python-all-dev being deprecated and unavailable in non standard Ubuntu installations. | Installing only PortAudio development headers fixes the problem: `sudo apt-get update` `sudo apt-get install portaudio19-dev`|

---

## Advanced Diagnostics (Community Tool)

For more advanced troubleshooting and automated health checks, you can use the community-built **OM1 Doctor** CLI tool:

https://github.com/CMZS4/om1-doctor

It provides:

- Disk & RAM diagnostic checks
- Python & environment validation
- Local port scan
- Log signature detection
- Markdown report generation

Example usage:

```bash
om1-doctor doctor
om1-doctor report --md-out report.md
```

The generated report can be attached to GitHub issues to help maintainers and users quickly resolve common node problems.
