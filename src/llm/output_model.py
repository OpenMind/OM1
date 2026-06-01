from typing import Optional
from pydantic import BaseModel, Field


class Action(BaseModel):
    type: str = Field(..., description="The specific type of action, such as 'move' or 'speak'")
    value: str = Field(..., description="The action argument")


class CortexOutputModel(BaseModel):
    actions: list[Action] = Field(..., description="List of actions to execute")
    thinking_duration: Optional[float] = Field(
        default=None,
        description="Optional duration in seconds to show thinking pose before executing actions"
    )