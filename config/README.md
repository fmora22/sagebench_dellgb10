# Configuration Guide

This directory contains configuration files for the project.

## Example Configuration Files

- `settings.json`: Application settings
- `environment.yml`: Environment configuration
- `config.ini`: Configuration parameters

## Usage

Configuration files should be loaded at application startup and can be environment-specific.

Example:
```python
import json

with open('config/settings.json', 'r') as f:
    config = json.load(f)
```
