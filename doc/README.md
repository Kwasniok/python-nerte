# Package Structure

![UML](/doc/uml_diagrams/nerte.svg)

UML Diagram: Simplified overview of the general package structure of NERTE inspired by domain-driven design.

|package|comment|
|:-|:-|
|nerte.values|fundamental (immutable) objects and constants|
|nerte.world|representation of objects in a world|
|nerte.geometry|representation of the geometry|
|nerte.render|render a world through the lens of a geometry|
|nerte.algorithm|generic algorithms|
|nerte.util|auxiliary components|


## Notes
- The UML diagrams are created with [DIA - Diagram Editor](https://en.wikipedia.org/wiki/Dia_%28software%29).
- __INFO__: The export option `SVG` is used since the `Cairo SVG` generates larger files.
- __HACK__: The gray background might be missing and can be added manually via `style="background-color: #7f7f7f"` in the `<svg>` tag.
