# Table of contents

## Get Started <a href="#developing" id="developing"></a>

* [OM1](README.md)
* [Introduction](developing/0_introduction.md)
* [Quick Start](developing/1_get-started.md)
* [Architecture](developing/2_architecture.md)

## Build with OM1 <a href="#build" id="build"></a>

* [Core Concepts](concepts.md)
    * [Project Structure](developing/7_project_structure.md)
    * [Configuration](developing/3_configuration.md)
    * [Inputs](developing/4_inputs.md)
    * [LLMs](developing/5_llms.md)
    * [Actions](developing/6_actions.md)
    * [Backgrounds](developing/8_backgrounds.md)
    * [Knowledge Base (RAG)](developing/knowledge_base.md)
    * [VAD & TTS Interrupt](developing/vad_tts_interrupt.md)
* Modes & Lifecycle
    * [Modes](full_autonomy_guidelines/modes.md)
    * [Mode Selection](full_autonomy_guidelines/mode_selection.md)
    * [Transition Rules](full_autonomy_guidelines/transition_rules.md)
    * [Lifecycle](full_autonomy_guidelines/lifecycle.md)
* [MCP Integration](developing/mcp-integration.md)
* [Middleware](developing/middleware.md)
    * [CycloneDDS](developing/cyclonedds.md)
    * [ROS2-humble](developing/ros2-humble.md)
    * [Zenoh ROS2 Bridge](developing/zenoh-bridge.md)
* Observability
    * [Metrics Reference](developing/metrics.md)
    * [Tracer & Quality Scorer](developing/tracer.md)
* [Troubleshooting Guide](developing/9_troubleshooting_guide.md)

## Developer Cookbook <a href="#developer-cookbook" id="developer-cookbook"></a>

* [Introduction](developer_cookbook/introduction.md)
    * [Configuration](developer_cookbook/config.md)
    * [Input](developer_cookbook/input.md)
    * [New Mode](developer_cookbook/new_mode.md)
* [Examples](examples/examples.md)
    * [Conversation](examples/conversation.md)
    * [Smart Toy & Companion](examples/smart_toy.md)

## Deploy Autonomy <a href="#full-autonomy-guidelines" id="full-autonomy-guidelines"></a>

* [Overview](full_autonomy_guidelines/architecture_overview.md)
* [BrainPack](full_autonomy_guidelines/brainpack_introduction.md)
* [API Overview](full_autonomy_guidelines/api_endpoints.md)
* [Autonomy Features](full_autonomy_guidelines/features/README.md)
    * [Machine Teleops](full_autonomy_guidelines/features/machine-teleops.md)
    * [Mapping & SLAM](full_autonomy_guidelines/features/mapping-slam.md)
    * [Hybrid Localisation](full_autonomy_guidelines/localization.md)
    * [Navigation (Nav2)](full_autonomy_guidelines/features/navigation.md)
    * [3D Map Navigation](full_autonomy_guidelines/features/3d-map-navigation.md)
    * [Frontier Exploration](full_autonomy_guidelines/features/frontier-exploration.md)
    * [Patrol](full_autonomy_guidelines/features/patrol.md)
    * [Auto Charging](full_autonomy_guidelines/features/auto-charging.md)
    * [Obstacle Avoidance](full_autonomy_guidelines/features/obstacle-avoidance.md)
    * [Person Following](full_autonomy_guidelines/features/person-following.md)
    * [Maps, Routes & Locations](full_autonomy_guidelines/features/maps-routes-locations.md)
    * [Memory Sync](full_autonomy_guidelines/features/memory-sync.md)
    * [Alerts](full_autonomy_guidelines/features/alerts.md)
    * [Video Recording](full_autonomy_guidelines/features/video-recording.md)
    * [Face Detection & Anonymization](full_autonomy_guidelines/features/face-detection-anonymization.md)
* [Plans & Access](developing/premium_features.md)

## Robots & Hardware <a href="#robots-hardware" id="robots-hardware"></a>

* [Unitree G1 Humanoid](robotics/unitree_g1_humanoid.md)
* [Unitree Go2 Quadruped](robotics/unitree_go2_quadruped.md)
* [Raspberry Pi](robotics/raspberrypi.md)
* [Tesla Dimo](robotics/tesla_dimo.md)
* [TurtleBot4](robotics/turtlebot4_zenoh.md)
* [UBTech Yanshee](robotics/ubtech_yanshee.md)
* [NVIDIA Thor](robotics/nvidia_thor.md)

## Simulators <a href="#simulators" id="simulators"></a>

* [Cloud Isaac Sim](simulators/cloud-isaac-sim.md)
* [Isaac Sim](simulators/isaac-sim.md)
* [Gazebo](simulators/gazebo.md)
* [Troubleshooting Guidelines](simulators/troubleshooting.md)

## API Reference <a href="#api-reference" id="api-reference"></a>

* [API Reference](api-reference/introduction.md)
    * [Account & Key Management](api-reference/endpoints/account_and_key_management.md)
    * [Google ASR](api-reference/endpoints/google_asr.md)
    * [ElevenLabs ASR](api-reference/endpoints/elevenlabs_asr.md)
    * [ElevenLabs TTS](api-reference/endpoints/elevenlabs_tts.md)
    * [LLM](api-reference/endpoints/llm.md)
    * [Riva](api-reference/endpoints/riva.md)
    * [ViLA VLM](api-reference/endpoints/vila_vlm.md)
* [Subscription Plans](api-reference/api_pricing.md)

## Release Notes <a href="#release-notes" id="release-notes"></a>

* Major Updates
  * [Beta Release](release-notes/major-updates/beta_release.md)
  * [Production Ready Release](release-notes/major-updates/production_ready_release.md)
* OM1
    * [beta](release-notes/om/beta.md)
    * [v1.0.x](release-notes/om/v1.0.x.md)
    * [Docker Images](release-notes/om/docker_images.md)
* OM1 Avatar
    * [beta](release-notes/om1-avatar/beta.md)
    * [v1.0.x](release-notes/om1-avatar/v1.0.x.md)
    * [Docker Images](release-notes/om1-avatar/docker_images.md)
* OM1 ROS2 SDK
    * [beta](release-notes/om1-ros2-sdk/beta.md)
    * [v1.0.x](release-notes/om1-ros2-sdk/v1.0.x.md)
    * [Docker Images](release-notes/om1-ros2-sdk/docker_images.md)
* Video Processor
    * [beta](release-notes/om1-video-processor/beta.md)
    * [v1.0.x](release-notes/om1-video-processor/v1.0.x.md)
    * [Docker Images](release-notes/om1-video-processor/docker_images.md)
* OM1 System Setup
    * [beta](release-notes/OM1-OTA/beta.md)
    * [v1.0.x](release-notes/OM1-OTA/v1.0.x.md)
    * [Docker Images](release-notes/OM1-OTA/docker_images.md)

## Good to Know <a href="#robotics" id="robotics"></a>

* [Asimov Governance](robotics/asimov_governance.md)
* [CRSF Long Range Control](robotics/crsf_long_range_control.md)
* [GPS Compass](robotics/gps_compass.md)
* [Mac](robotics/mac.md)
* [Media Server](robotics/Media_server.md)
* [Motion Planning LiDAR A1M8](robotics/motion_planning_lidarA1M8.md)
* [Motion Planning TurtleBot4](robotics/motion_planning_turtlebot4.md)
* [Motion Planning Unitree Go2](robotics/motion_planning_unitree_go2.md)
* [RF Mapping](robotics/rf_mapping.md)
* [Unitree Go2 Quadruped Configurations](robotics/unitree_go2_quadruped_configurations.md)
* [Zenoh](robotics/zenoh.md)
