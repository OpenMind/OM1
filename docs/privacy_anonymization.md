# Real-Time Data Anonymization API for OM1

## Overview
This module provides simple anonymization of personal or sensitive information in real-time.

## Usage
```python
from om1.privacy.anonymizer import Anonymizer

anon = Anonymizer()
text = "User john.doe@gmail.com paid with card 4242-4242-4242-4242"
print(anon.anonymize(text))
```

## Configuration Example
```json5
{
  "privacy_settings": {
    "location_data": {
      "method": "coarse_graining",
      "granularity_meters": 50
    },
    "timestamp_data": {
      "method": "fuzzing",
      "range_seconds": 60
    }
  }
}
```

- If privacy is enabled in agent config, OM1 runtime will automatically route data through this service.
