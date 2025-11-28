#!/bin/bash

clear
echo "========================================"
echo "  Priority Task Queue Plugin - Demo"
echo "  Issue #629 - OM1 Agent Coordination"
echo "========================================"
echo ""
sleep 2

echo "Running test suite..."
echo ""
python3 -m pytest tests/test_task_queue.py -v -p no:rostest
echo ""
sleep 3

echo "Running interactive demo..."
echo ""
python3 demo_task_queue.py
