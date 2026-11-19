# Layers

The conservation layers a model steps. Concepts: [Layers](../../concepts/layers.md); options: [Kirchhoff](../kirchhoff-options.md), [transport](../transport-options.md), [Model](../model-options.md).

## Flow by potential

::: noodl.layers.kirchhoff
    options:
      members:
        - KirchhoffLayer

## Transport

::: noodl.layers.transport
    options:
      members:
        - TransportLayer
        - active_interior
        - UndefinedSteadyState
        - TransportStep
        - ExpmResult

## Reactions

::: noodl.layers.reaction
    options:
      members:
        - Reaction

## Flow by allocation

::: noodl.layers.allocation
    options:
      members:
        - AllocatedFlowLayer
