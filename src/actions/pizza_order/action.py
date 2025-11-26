from actions.base import AgentAction, ActionConfig
from actions.pizza_order.interface import PizzaOrderInput
from actions.pizza_order.connector.create_order import CreateOrderConnector

class PizzaOrder(AgentAction):
    def __init__(self):
        config = ActionConfig(name="pizza_order")
        super().__init__(
            name="pizza_order",
            llm_label="Pizza Order",         # label for prompts/logs
            interface=PizzaOrderInput,
            connector=CreateOrderConnector(config),
            exclude_from_prompt=False
        )
