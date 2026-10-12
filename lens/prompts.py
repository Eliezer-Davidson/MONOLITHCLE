# MONOLITHCLE (package `lens`). External components: THIRD_PARTY_NOTICES.md.
# All prompt text in this file was written for this project.
"""Prompts for the scenegraph stage. Contract: LIST_OBJECTS returns one line
per object, `<short name> | <category> | <supported by>`; floor is implied."""

LIST_OBJECTS = """\
This photo shows a real indoor space that will be rebuilt inside a physics
simulator for robot training. List every distinct movable object and every
piece of furniture that a robot could touch, bump into, or place things on.

Rules:
- One line per object: <short unique name> | <one-word category> | <what it rests on>
- <what it rests on> is another listed object, or 'floor'.
- Name colors/materials in the short name so similar objects stay distinguishable.
- Skip: walls, ceiling, windows, doors, lights, and anything painted on or
  attached to a wall. Skip reflections and things seen through glass.
- If two identical objects appear, give each its own line and number them.
Answer with the lines only, no extra text."""

def render_verdict_prompt(name: str) -> str:
    return (f"The second image highlights a region of the first image. Does the "
            f"highlighted region cover exactly the object '{name}' — the whole "
            f"object and nothing else? Reply with one word: YES or NO.")
