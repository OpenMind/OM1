    g_ut_eth = config.get("unitree_ethernet")
    g_URID = config.get("URID")
    g_robot_ip = config.get("robot_ip")

    backgrounds = [
        load_background(bg["type"])(
            config=BackgroundConfig(
                **add_meta(bg.get("config", {}), api_key, g_ut_eth, g_URID, g_robot_ip)
            )
        )
        for bg in config.get("backgrounds", [])
    ]
    agent_inputs = [
        load_input(inp["type"])(
            config=SensorConfig(
                **add_meta(inp.get("config", {}), api_key, g_ut_eth, g_URID, g_robot_ip)
            )
        )
        for inp in config.get("agent_inputs", [])
    ]
    simulators = [
        load_simulator(sim["type"])(
            config=SimulatorConfig(
                name=sim["type"],
                **add_meta(
                    sim.get("config", {}), api_key, g_ut_eth, g_URID, g_robot_ip
                ),
            )
        )
        for sim in config.get("simulators", [])
    ]
    agent_actions = [
        load_action(
            {
                **action,
                "config": add_meta(
                    action.get("config", {}), api_key, g_ut_eth, g_URID, g_robot_ip
                ),
            }
        )
        for action in config.get("agent_actions", [])
    ]
    cortex_llm = load_llm(config["cortex_llm"]["type"])(
        config=LLMConfig(
            **add_meta(  # type: ignore
                config["cortex_llm"].get("config", {}),
                api_key,
                g_ut_eth,
                g_URID,
                g_robot_ip,
            )
        ),
        available_actions=agent_actions,
    )
    return RuntimeConfig(
        version=config.get("version", "v1.0.0"),  # Default version if not specified
        hertz=config.get("hertz", 1),
        name=config.get("name", "TestAgent"),
        system_prompt_base=config.get("system_prompt_base", ""),
        system_governance=config.get("system_governance", ""),
        system_prompt_examples=config.get("system_prompt_examples", ""),
        agent_inputs=agent_inputs,
        cortex_llm=cortex_llm,
        simulators=simulators,
        agent_actions=agent_actions,
        backgrounds=backgrounds,
    )
