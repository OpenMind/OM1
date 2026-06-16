---
title: Development Guide
description: "Introduction"
icon: brain
---

The Developer Cookbook is a collection of practical, high-impact recipes designed to help you extend, customize, and build on top of the OpenMind platform.
If the Quickstart shows you how to use OM1, this Cookbook shows you how to build with it.

Here's what you can do with OM1
1. Build a new config file
2. Introduce a new mode
3. Configure a new Input Plugin

Before building with OM1, make sure you've completed the [Getting Started](../developing/1_get-started) guide and have OM1 installed. Understand the important concepts and components that are part of OM1.

Then dive into any recipe that interests you!

## Development workflow

### Linting and Testing (Mandatory)

Run repository checks before committing:

```bash
make check
```

Or run individual checks when needed:

```bash
make fmt
make lint
make test
```

### Unit Testing

To unit test the system, run:
```bash
make test
```

Use clear naming conventions and comments for better code maintainability.
