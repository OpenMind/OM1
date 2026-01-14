import logging
from actions.base import ActionConfig, ActionConnector
from actions.process_payment.interface import PaymentInput

class ConsoleConnector(ActionConnector[PaymentInput]):
    def __init__(self, config: ActionConfig):
        super().__init__(config)

    async def connect(self, output_interface: PaymentInput) -> None:
        print(output_interface)
