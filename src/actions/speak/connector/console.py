from actions.base import ActionConfig, ActionConnector
from actions.speak.interface import SpeakInput


class ConsoleConnector(ActionConnector[SpeakInput]):
    """
    Console connector for speak action.
    """

    def __init__(self, config: ActionConfig):
        super().__init__(config)

    async def connect(self, input_data: SpeakInput) -> SpeakInput:
        """
        Print the text to the console.
        """
        print(f"\n[OM1 Assistant] {input_data.action}\n")
        return input_data
