"""Init command - Create new configuration files."""

from typing import Optional

import typer

from cli.utils.config import get_config_dir
from cli.utils.output import console, print_error, print_info, print_success

BASIC_TEMPLATE = """{{
  version: "v1.0.1",
  hertz: 1,
  name: "{name}",
  api_key: "openmind_free",
  URID: "om1_{name}",
  system_prompt_base: "You are a helpful AI assistant.",
  system_governance: "Here are the laws that govern your actions. Do not violate these laws.\\nFirst Law: A robot cannot harm a human or allow a human to come to harm.\\nSecond Law: A robot must obey orders from humans, unless those orders conflict with the First Law.\\nThird Law: A robot must protect itself, as long as that protection doesn't conflict with the First or Second Law.\\nThe First Law is considered the most important, taking precedence over the second and third laws.",
  system_prompt_examples: "",
  agent_inputs: [
    // Add your inputs here
    // {{
    //   type: "VLM_Local_YOLO",
    //   config: {{
    //     camera_index: 0,
    //   }},
    // }},
  ],
  cortex_llm: {{
    type: "OpenAILLM",
    config: {{
      agent_name: "{display_name}",
      history_length: 10,
    }},
  }},
  agent_actions: [
    // Add your actions here
    // {{
    //   name: "speak",
    //   llm_label: "speak",
    //   connector: "elevenlabs_tts",
    // }},
  ],
  backgrounds: [
    // Add background processes here
  ],
}}
"""

CONVERSATION_TEMPLATE = """{{
  version: "v1.0.1",
  hertz: 1,
  name: "{name}",
  api_key: "openmind_free",
  URID: "om1_{name}",
  system_prompt_base: "You are a friendly conversational AI. Engage in natural dialogue, answer questions thoughtfully, and be helpful.",
  system_governance: "Here are the laws that govern your actions. Do not violate these laws.\\nFirst Law: A robot cannot harm a human or allow a human to come to harm.\\nSecond Law: A robot must obey orders from humans, unless those orders conflict with the First Law.\\nThird Law: A robot must protect itself, as long as that protection doesn't conflict with the First or Second Law.",
  system_prompt_examples: "",
  agent_inputs: [
    {{
      type: "GoogleASRInput",
    }},
  ],
  cortex_llm: {{
    type: "OpenAILLM",
    config: {{
      agent_name: "{display_name}",
      history_length: 20,
    }},
  }},
  agent_actions: [
    {{
      name: "speak",
      llm_label: "speak",
      connector: "elevenlabs_tts",
      config: {{
        voice_id: "TbMNBJ27fH2U0VgpSNko",
      }},
    }},
  ],
  backgrounds: [],
}}
"""

ROBOT_TEMPLATE = """{{
  version: "v1.0.1",
  hertz: 1,
  name: "{name}",
  api_key: "openmind_free",
  URID: "om1_{name}",
  system_prompt_base: "You are a robot assistant. Use your sensors to understand the environment and take appropriate actions.",
  system_governance: "Here are the laws that govern your actions. Do not violate these laws.\\nFirst Law: A robot cannot harm a human or allow a human to come to harm.\\nSecond Law: A robot must obey orders from humans, unless those orders conflict with the First Law.\\nThird Law: A robot must protect itself, as long as that protection doesn't conflict with the First or Second Law.",
  system_prompt_examples: "",
  agent_inputs: [
    {{
      type: "VLM_Local_YOLO",
      config: {{
        camera_index: 0,
        log_file: true,
      }},
    }},
  ],
  cortex_llm: {{
    type: "OpenAILLM",
    config: {{
      agent_name: "{display_name}",
      history_length: 10,
    }},
  }},
  agent_actions: [
    {{
      name: "speak",
      llm_label: "speak",
      connector: "elevenlabs_tts",
      config: {{
        voice_id: "TbMNBJ27fH2U0VgpSNko",
      }},
    }},
    // Add robot-specific actions here
    // {{
    //   name: "move_go2_autonomy",
    //   llm_label: "move",
    //   connector: "unitree_sdk",
    // }},
  ],
  backgrounds: [],
}}
"""

MODE_AWARE_TEMPLATE = """{{
  version: "v1.0.1",
  default_mode: "default",
  allow_manual_switching: true,
  mode_memory_enabled: true,

  // Global settings
  api_key: "openmind_free",
  system_governance: "Here are the laws that govern your actions. Do not violate these laws.\\nFirst Law: A robot cannot harm a human or allow a human to come to harm.\\nSecond Law: A robot must obey orders from humans, unless those orders conflict with the First Law.\\nThird Law: A robot must protect itself, as long as that protection doesn't conflict with the First or Second Law.",
  cortex_llm: {{
    type: "OpenAILLM",
    config: {{
      agent_name: "{display_name}",
      history_length: 10,
    }},
  }},

  modes: {{
    default: {{
      display_name: "Default Mode",
      description: "Standard operating mode",
      system_prompt_base: "You are {display_name} in default mode. Be helpful and responsive.",
      hertz: 1,
      agent_inputs: [
        {{
          type: "GoogleASRInput",
        }},
      ],
      agent_actions: [
        {{
          name: "speak",
          llm_label: "speak",
          connector: "elevenlabs_tts",
          config: {{
            voice_id: "TbMNBJ27fH2U0VgpSNko",
          }},
        }},
      ],
    }},

    active: {{
      display_name: "Active Mode",
      description: "High activity mode with more inputs",
      system_prompt_base: "You are {display_name} in active mode. Focus on the task at hand.",
      hertz: 2,
      agent_inputs: [
        {{
          type: "GoogleASRInput",
        }},
        {{
          type: "VLM_Local_YOLO",
          config: {{
            camera_index: 0,
          }},
        }},
      ],
      agent_actions: [
        {{
          name: "speak",
          llm_label: "speak",
          connector: "elevenlabs_tts",
          config: {{
            voice_id: "TbMNBJ27fH2U0VgpSNko",
          }},
        }},
      ],
    }},
  }},

  transition_rules: [
    {{
      from_mode: "default",
      to_mode: "active",
      transition_type: "input_triggered",
      trigger_keywords: ["activate", "start", "go"],
      priority: 1,
      cooldown_seconds: 5.0,
    }},
    {{
      from_mode: "active",
      to_mode: "default",
      transition_type: "input_triggered",
      trigger_keywords: ["stop", "pause", "rest"],
      priority: 1,
      cooldown_seconds: 5.0,
    }},
    {{
      from_mode: "*",
      to_mode: "default",
      transition_type: "input_triggered",
      trigger_keywords: ["reset", "default mode"],
      priority: 5,
      cooldown_seconds: 10.0,
    }},
  ],
}}
"""


TEMPLATES = {
    "basic": ("Basic configuration", BASIC_TEMPLATE),
    "conversation": ("Conversation-focused configuration", CONVERSATION_TEMPLATE),
    "robot": ("Robot configuration with sensors", ROBOT_TEMPLATE),
    "mode-aware": ("Multi-mode configuration", MODE_AWARE_TEMPLATE),
}


def init(
    name: str = typer.Argument(
        None,
        help="Name for the new configuration (without .json5 extension).",
    ),
    template: str = typer.Option(
        "basic",
        "--template",
        "-t",
        help="Template to use: basic, conversation, robot, mode-aware.",
    ),
    display_name: Optional[str] = typer.Option(
        None,
        "--display-name",
        "-d",
        help="Display name for the agent. Defaults to capitalized config name.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Overwrite existing configuration.",
    ),
    list_templates: bool = typer.Option(
        False,
        "--list-templates",
        "-l",
        help="List available templates.",
    ),
) -> None:
    """
    Create a new OM1 configuration file.

    Generates a configuration file from a template with common defaults
    and examples.

    Examples
    --------
        om1 init mybot                      # Create basic config
        om1 init mybot --template robot     # Create robot config
        om1 init mybot --template mode-aware # Create multi-mode config
        om1 init mybot -d "My Robot"        # Set display name
        om1 init --list-templates           # Show available templates
    """
    if list_templates:
        _list_templates()
        return

    if name is None:
        print_error("Please provide a configuration name.")
        print_info("Usage: om1 init <name> [--template <template>]")
        raise typer.Exit(1)

    if template not in TEMPLATES:
        print_error(f"Unknown template: {template}")
        print_info(f"Available templates: {', '.join(TEMPLATES.keys())}")
        raise typer.Exit(1)

    config_dir = get_config_dir()
    config_path = config_dir / f"{name}.json5"

    if config_path.exists() and not force:
        print_error(f"Configuration already exists: {name}.json5")
        print_info("Use --force to overwrite.")
        raise typer.Exit(1)

    if not display_name:
        display_name = name.replace("_", " ").replace("-", " ").title()

    _, template_content = TEMPLATES[template]
    content = template_content.format(name=name, display_name=display_name)

    try:
        config_path.write_text(content, encoding="utf-8")
        print_success(f"Created configuration: {config_path}")
        console.print()
        console.print("[bold]Next steps:[/bold]")
        console.print(f"  1. Edit your configuration: [cyan]{config_path}[/cyan]")
        console.print(f"  2. Validate it: [cyan]om1 validate {name}[/cyan]")
        console.print(f"  3. Run it: [cyan]om1 run {name}[/cyan]")

    except Exception as e:
        print_error(f"Failed to create configuration: {e}")
        raise typer.Exit(1)


def _list_templates() -> None:
    """List available templates."""
    console.print("[bold]Available templates:[/bold]\n")
    for template_name, (description, _) in TEMPLATES.items():
        console.print(f"  [cyan]{template_name:15}[/cyan] {description}")
    console.print()
    console.print("Usage: [dim]om1 init <name> --template <template>[/dim]")
