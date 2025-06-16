# Ironhide

Ironhide simplify the process of building high-quality autonomous agents by representing them as classes.

## Install

```
uv add git+https://github.com/bytewer-lab/ironhide.git
```

## Features

- OOP Abstraction
    - The Ironhide core is `BaseAgent` abstract class that convert the child class into a Agent.
- Static Typing
    - All the source code is based on Mypy strict, Ruff and Pydantic.
- Dependency Injection
    -  The child class constructor can receive extra arguments
- Structured Output
    - The agent can have a global or per message output_format
- Chain of Thought
    - Is possible to define reasoning steps before call the methods
- Auto Function Calling
    - Automatically extract the methods information and convert it tools
- Feedback Loop
    - After call the tools, is possible to evaluate the previous behavior and aprove or reject it

## Limitations
1. Gemini don't support use structured_output with function calling at the same request
1. Deepseek don't support structured output