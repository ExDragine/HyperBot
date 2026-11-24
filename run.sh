#!/usr/bin/env bash

source .venv/bin/activate

while true; do
    uv run --locked nb run
    sleep 5
done
