#!/bin/bash

# Start Riva in background
start-riva --riva-uri=0.0.0.0:${riva_speech_api_port:-50051} \
          --asr_service=${service_enabled_asr:-true} \
          --tts_service=${service_enabled_tts:-true} \
          --nlp_service=${service_enabled_nlp:-true} &

# Wait for port
while ! nc -z localhost 50051; do sleep 1; done;
echo "Port 50051 is ready. "

# Run Python script
echo "Starting Python scripts..."
cd /app
python3 -m om1_speech ${PYTHON_ARGS}