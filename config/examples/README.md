# Robot Character Library

This directory contains several pre-configured robot characters to help you get started with customizing your robot's personality and behavior.

## Available Characters

*   **Sentinel (security_guard.json5)**: A professional and vigilant security robot.
*   **Curator (museum_guide.json5)**: An elegant and knowledgeable museum guide.
*   **Jeeves (home_assistant.json5)**: A polite and efficient home assistant.
*   **Spark (education_robot.json5)**: An enthusiastic and patient education robot.

## How to Create Your Own Character

To create a custom character, you can copy one of these examples and modify the `system_prompt_base` and `system_prompt_examples` fields.

### Best Practices

1.  **Define a clear role**: Start the prompt with "You are [Character Name], a [Role]...".
2.  **Establish tone**: Describe how the character should speak (e.g., "professional," "witty," "enthusiastic").
3.  **Provide examples**: Use the `system_prompt_examples` field to show how the character should respond to specific situations.
4.  **Keep it concise**: For action-oriented robots, use brief and direct language.
