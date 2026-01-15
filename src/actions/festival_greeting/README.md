# Festival Greeting Skill

## Overview

The Festival Greeting and Reminder Skill is a human-robot interaction feature for OM1 robots that allows robots to send greetings and reminders to users on various festivals and special occasions.

## Features

1. **Festival Greetings**: Supports multiple festival types (Chinese New Year, Mid-Autumn Festival, Christmas, New Year, etc.)
2. **Personalized Greetings**: Can specify recipient names for personalized greetings
3. **Festival Reminders**: Automatically detects upcoming festivals and reminds users
4. **Custom Messages**: Supports custom greeting messages
5. **Multi-language Support**: Supports festival names and greetings in multiple languages

## Architecture

### 1. Action Interface (`interface.py`)

Defines the `FestivalGreeting` action interface, including:
- `FestivalType`: Festival type enumeration (Chinese New Year, Mid-Autumn Festival, Christmas, etc.)
- `FestivalGreetingInput`: Input interface containing festival type, message, and recipient name
- `FestivalGreeting`: Action interface definition

### 2. Connector (`connector/elevenlabs_tts.py`)

Implements voice greeting functionality through ElevenLabs TTS:
- Supports ElevenLabs TTS configuration
- Automatically generates default greeting messages
- Supports personalized messages and recipient names
- Integrates with conversation history

### 3. Festival Provider (`providers/festival_provider.py`)

Festival data management module:
- Maintains festival calendar
- Detects today's festivals
- Detects upcoming festivals
- Supports custom festival addition
- Manages reminder time configuration

### 4. Background Task (`backgrounds/plugins/festival_reminder.py`)

Background task that periodically checks festivals and updates context:
- Periodically checks festivals (default: every hour)
- Updates system context so LLM can be aware of festival information
- Supports configurable check intervals and reminder times

## Usage

### 1. Configuration File Example

Create a configuration file (e.g., `festival_greeting_example.json5`) containing:

```json5
{
  name: "festival_greeting",
  agent_actions: [
    {
      name: "festival_greeting",
      llm_label: "festival_greeting",
      connector: "elevenlabs_tts",
      config: {
        elevenlabs_api_key: "your_key",
        voice_id: "JBFqnCBsd6RMkjVDRZzb",
      },
    },
  ],
  agent_backgrounds: [
    {
      type: "FestivalReminder",
      config: {
        check_interval_seconds: 3600,
        enable_reminders: true,
      },
    },
  ],
}
```

### 2. Running

```bash
uv run src/run.py festival_greeting
```

### 3. LLM Usage Examples

The LLM can trigger festival greetings in the following ways:

```python
# Chinese New Year greeting
FestivalGreeting: {
  'festival_type': 'chinese_new_year',
  'message': 'Happy Chinese New Year! Wishing you good health and all the best!'
}

# Personalized birthday greeting
FestivalGreeting: {
  'festival_type': 'birthday',
  'recipient_name': 'Alice',
  'message': 'Happy Birthday!'
}

# Festival reminder
FestivalGreeting: {
  'festival_type': 'mid_autumn',
  'message': 'Mid-Autumn Festival is coming in 3 days, don't forget to prepare mooncakes!'
}
```

## Supported Festival Types

- `chinese_new_year`: Chinese New Year
- `mid_autumn`: Mid-Autumn Festival
- `dragon_boat`: Dragon Boat Festival
- `national_day`: National Day
- `christmas`: Christmas
- `new_year`: New Year
- `valentine`: Valentine's Day
- `birthday`: Birthday
- `custom`: Custom festival

## Extension Suggestions

1. **Lunar Calendar Support**: Add lunar date calculation for accurate support of traditional Chinese festivals
2. **Multi-language Greetings**: Extend support for default greeting messages in more languages
3. **Festival Expressions**: Combine with Face action to display appropriate expressions during greetings
4. **Festival Music**: Play festival-related background music during greetings
5. **User Preferences**: Record user preferences for festivals and greeting styles
6. **Reminder Strategies**: Support more flexible reminder strategies (email, SMS, etc.)

## Notes

1. Festival dates currently use the Gregorian calendar; traditional Chinese festivals require manual date updates
2. It is recommended to integrate a lunar calendar library (such as `zhdate`) for accurate calculation of lunar festivals
3. ElevenLabs API key needs to be configured to use TTS functionality
4. Background task needs to be properly registered to run in the background
