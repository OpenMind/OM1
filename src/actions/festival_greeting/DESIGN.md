# Festival Greeting and Reminder Skill Design Document

## Design Philosophy

Based on your idea, I've designed a complete festival greeting and reminder skill with the following core features:

1. **Proactive Greetings**: Robot actively sends greetings on festival days
2. **Advance Reminders**: Reminds users before festivals arrive (e.g., 7 days, 3 days, 1 day in advance)
3. **Personalized Greetings**: Supports specifying recipient names for personalized greetings
4. **Multi-festival Support**: Supports major Chinese and Western festivals

## Architecture Design

### 1. Modular Design

```
festival_greeting/
├── interface.py              # Action interface definition
├── connector/
│   └── elevenlabs_tts.py    # TTS connector implementation
└── README.md                 # Usage documentation

providers/
└── festival_provider.py      # Festival data management

backgrounds/plugins/
└── festival_reminder.py      # Background reminder task
```

### 2. Data Flow

```
User/LLM Trigger
    ↓
FestivalGreeting Action
    ↓
ElevenLabs TTS Connector
    ↓
Voice Output

Background Task (FestivalReminder)
    ↓
FestivalProvider (Check Festivals)
    ↓
ContextProvider (Update Context)
    ↓
LLM Perceives Festival Information
    ↓
Auto-trigger Greetings
```

## Core Functionality Implementation

### 1. FestivalGreeting Action

**Interface Design**:
- `festival_type`: Festival type (enum value)
- `message`: Custom greeting message (optional)
- `recipient_name`: Recipient name (optional)

**Features**:
- Supports 9 festival types
- Automatically generates default greetings
- Supports personalization

### 2. FestivalProvider

**Functionality**:
- Maintains festival calendar
- Detects today's festivals
- Detects upcoming festivals (configurable days)
- Manages reminder time points (7 days, 3 days, 1 day before)

**Extensibility**:
- Supports adding custom festivals
- Supports configuring reminder time points

### 3. FestivalReminder Background

**Functionality**:
- Periodically checks festivals (default: every hour)
- Updates system context so LLM can be aware of festival information
- Supports configurable check intervals and reminder times

## Usage Recommendations

### 1. System Prompt Design

In the configuration file's `system_prompt_base`, it's recommended to add:

```
"You are a friendly and considerate robot assistant. You are aware of festivals 
and holidays, and you can send warm greetings to users. When you notice that 
today is a festival or a festival is approaching, you should proactively send 
greetings using the festival_greeting action."
```

### 2. Context Awareness

The background task updates context, and the LLM can access festival information through:

- `today_festivals`: List of today's festivals
- `upcoming_festivals`: Upcoming festivals (within 7 days)
- `reminder_festivals`: Festivals that need reminders

### 3. Integration with Other Actions

Can be combined with other actions to enhance the experience:

```python
# Festival greeting + expression
FestivalGreeting: {'festival_type': 'chinese_new_year'}
Face: {'action': 'happy'}

# Festival greeting + voice + movement
FestivalGreeting: {'festival_type': 'birthday', 'recipient_name': 'Alice'}
Speak: {'action': 'Let's celebrate together!'}
Move: {'action': 'dance'}
```

## Extension Suggestions

### 1. Lunar Calendar Support (Important)

The current implementation uses Gregorian dates, which is not accurate for traditional Chinese festivals. Recommendation:

```python
# Install lunar calendar library
pip install zhdate

# Use in FestivalProvider
from zhdate import ZhDate

# Calculate lunar festivals
chinese_new_year = ZhDate(2025, 1, 1).to_datetime().date()
```

### 2. Multi-language Support

Extend default greeting messages to support more languages:

```python
messages = {
    "chinese_new_year": {
        "zh": "新年快乐！",
        "en": "Happy Chinese New Year!",
        "ja": "明けましておめでとうございます！"
    }
}
```

### 3. User Preference Learning

Record user preferences for different festivals:

```python
user_preferences = {
    "alice": {
        "favorite_festivals": ["birthday", "christmas"],
        "greeting_style": "formal"
    }
}
```

### 4. Festival Expression Mapping

Automatically select appropriate facial expressions for different festivals:

```python
festival_emotions = {
    "chinese_new_year": "excited",
    "birthday": "happy",
    "valentine": "curious"
}
```

### 5. Reminder Strategy Optimization

- Support multiple reminder methods (voice, text, email)
- Support user-customized reminder times
- Support repeated reminders (e.g., daily reminders)

### 6. Festival Knowledge Base

Extend festival information, including:
- Festival historical background
- Festival customs
- Festival-related activity suggestions

## Testing Recommendations

1. **Unit Tests**: Test FestivalProvider's date calculation logic
2. **Integration Tests**: Test complete flow of Action + Connector
3. **Scenario Tests**: Test greeting effects for different festival types
4. **Performance Tests**: Test performance impact of background task

## Notes

1. **Timezone Handling**: Ensure festival date calculations consider timezones
2. **API Keys**: Need to configure ElevenLabs API key
3. **Resource Consumption**: Background task runs periodically, be mindful of resource consumption
4. **Error Handling**: Network errors, API errors, etc. need proper handling

## Summary

This design implements the festival greeting and reminder functionality you proposed, with the following advantages:

✅ **Modular**: Easy to maintain and extend  
✅ **Intelligent**: LLM can automatically perceive festivals and trigger greetings  
✅ **Personalized**: Supports custom messages and recipients  
✅ **Extensible**: Easy to add new festivals and features  

I hope this design meets your needs! If you have any questions or need adjustments, please let me know.
