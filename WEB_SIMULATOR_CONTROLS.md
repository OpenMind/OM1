# How to Control the Robot in the Browser

The Three.js simulator now includes a control panel at the bottom of the screen with buttons to move the robot.

## Control Buttons

At the bottom of the browser window, you'll see 5 buttons:

1. **↶ Turn Left** - Rotates the robot 90° counterclockwise
2. **↑ Move Forward** - Moves the robot forward 0.5 meters
3. **↷ Turn Right** - Rotates the robot 90° clockwise  
4. **↓ Move Back** - Moves the robot backward 0.5 meters
5. **⏸ Stop** - Stops all movement

## How It Works

When you click a button, the simulator:

1. **First tries** to send the command via MockInput WebSocket (port 8765)
   - This sends the command through OM1's LLM for natural language processing
   - The LLM interprets the command and sends it to the action connector
   - This is the recommended way as it goes through the full OM1 pipeline

2. **Falls back** to direct API call if MockInput is not available
   - Sends HTTP POST to `/api/command` endpoint
   - Updates robot state directly in the simulator
   - Useful for quick testing without LLM processing

## Alternative Methods

### Method 1: Via MockInput WebSocket (Recommended)

Connect to the MockInput WebSocket server and send text commands:

```javascript
const ws = new WebSocket('ws://localhost:8765');
ws.onopen = () => {
    ws.send('move forward');
    ws.send('turn left');
    ws.send('stop');
};
```

### Method 2: Via HTTP API

Send POST requests directly to the simulator:

```bash
curl -X POST http://localhost:8001/api/command \
  -H "Content-Type: application/json" \
  -d '{"action": "move forwards"}'
```

### Method 3: Natural Language via MockInput

Send natural language commands that the LLM will interpret:

```javascript
const ws = new WebSocket('ws://localhost:8765');
ws.send('Please move the robot forward and then turn left');
```

## Keyboard Shortcuts (Future Enhancement)

You can add keyboard shortcuts by modifying the HTML. For example:

```javascript
document.addEventListener('keydown', (e) => {
    switch(e.key) {
        case 'ArrowLeft': sendCommand('turn left'); break;
        case 'ArrowRight': sendCommand('turn right'); break;
        case 'ArrowUp': sendCommand('move forwards'); break;
        case 'ArrowDown': sendCommand('move back'); break;
        case ' ': sendCommand('stand still'); break;
    }
});
```

## Troubleshooting

**Buttons don't work:**
- Check browser console for errors
- Verify WebSocket connection status (shown in info panel)
- Try refreshing the page

**Robot doesn't move:**
- Check that OM1 is running with the web_sim config
- Verify the action connector is configured correctly
- Look at OM1 logs for error messages

**Commands go through but robot doesn't respond:**
- The command might be going through LLM processing (check OM1 logs)
- Try using direct API method for immediate response
- Verify robot state updates in the info panel

