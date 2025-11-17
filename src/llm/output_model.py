from pydantic import BaseModel, Field


class Action(BaseModel):
    """
    Executable action with its argument.

    Parameters
    ----------
    type : str
        Type of action to execute, such as 'move' or 'speak'
    value : str
        The action argument, such as the magnitude of a movement or the sentence to speak
    """

    type: str = Field(..., description="The specific type of action, such as 'move' or 'speak'")
    value: str = Field(..., description="The action argument")


class CortexOutputModel(BaseModel):
    """
    Output model for the Cortex LLM responses.

    Parameters
    ----------
    actions : list[Action]
        List of actions to be executed
    """

    actions: list[Action] = Field(..., description="List of actions to execute")
