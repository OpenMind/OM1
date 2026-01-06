# Hot-Reload Configuration

Real-time config updates without restart.

## Usage
\`\`\`python
class MyAgent(HotReloadMixin):
    def __init__(self):
        super().__init__(hot_reload_enabled=True)
        self.setup_hot_reload("my_agent")
\`\`\`

Edit config file → changes apply instantly.
