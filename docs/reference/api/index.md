# API reference

Generated from the package's docstrings, one page per package. Start with the objects below: they
are enough to build, check and run a model. For a one-line description of every public name, use the
[Catalogue](../catalogue/index.md); for the ideas behind these objects, [Concepts](../../concepts/index.md).

## Start here

| Object | What it is | Page |
|---|---|---|
| [`Network`][noodl.topology.Network] | The directed multigraph of nodes and typed edges every model is built on. | [Topology and networks](topology.md) |
| [`KirchhoffLayer`][noodl.layers.kirchhoff.KirchhoffLayer] | A layer whose flows are driven by a potential (pressure, head, voltage). | [Layers](layers.md) |
| [`TransportLayer`][noodl.layers.transport.TransportLayer] | A layer that carries a quantity (heat, species, charge) along the flows. | [Layers](layers.md) |
| [`Model`][noodl.model.Model] | Layers on one network, stepped together. | [Model and coupling](model.md) |
| [`drivers_from`][noodl.refs.drivers_from], [`state_from`][noodl.refs.state_from] | Build drivers and state by label rather than by index. | [Model and coupling](model.md) |
| [`check_setup`][noodl.validation.check_setup] | Check a model, its state and its drivers before running. | [Model and coupling](model.md) |
| [`union`][noodl.couple.union] | Couple several models into one. | [Model and coupling](model.md) |

## Build a model for an application

Each application provides a `build_model` that returns the model, its initial state and its driver
template, and readers for the files of the tool it mirrors.

| Application | Builder | Readers |
|---|---|---|
| [Building physics](building.md) | [`build_model`][noodl.apps.building_physics.thermal.build_model] | [`read_prj`][noodl.apps.building_physics.prj.read_prj], [`read_wth`][noodl.apps.building_physics.wth.read_wth], [`read_epw`][noodl.apps.building_physics.epw.read_epw], [`read_modelica`][noodl.apps.building_physics.modelica.read_modelica] |
| [Street air quality](street.md) | [`build_model`][noodl.apps.street_aq.network.build_model] | [`read_case`][noodl.apps.street_aq.case.read_case] |
| [Sewers](sewer.md) | [`build_model`][noodl.apps.sewer.network.build_model] | [`read_swmm_inp`][noodl.apps.sewer.inp.read_swmm_inp] |
| [Water distribution](water.md) | [`build_model`][noodl.apps.water.network.build_model] | [`read_epanet_inp`][noodl.apps.water.inp.read_epanet_inp] |
| [Water resources](water-resources.md) | [`build_model`][noodl.apps.wsimod.build.build_model] | [`read_wsimod_config`][noodl.apps.wsimod.config.read_wsimod_config] |
| [Electrical circuits](electrical.md) | [`build_model`][noodl.apps.electrical.circuit.build_model] | [`read_modelica_circuit`][noodl.apps.electrical.modelica.read_modelica_circuit], [`read_matpower`][noodl.apps.electrical.power.matpower.read_matpower] |

The formats each reader accepts are in [File formats](../formats.md).

## Pages

| Page | Contents |
|---|---|
| [Topology and networks](topology.md) | `Network`, cycle and flow-basis helpers. |
| [Constitutive laws and heads](laws.md) | The general-purpose laws, heads, withdrawals and storage, and the base classes to extend. |
| [Layers](layers.md) | `KirchhoffLayer`, `TransportLayer`, `Reaction`, `AllocatedFlowLayer`. |
| [Model and coupling](model.md) | `Model`, closure contracts, set-up helpers, validation, coupling. |
| [Components](components.md) | Reusable sub-networks. |
| [Solvers and operators](solvers.md) | Linear operators, Newton, implicit differentiation, iterative solvers. |
| [Building physics](building.md), [Street air quality](street.md), [Sewers](sewer.md), [Water distribution](water.md), [Water resources](water-resources.md), [Electrical circuits](electrical.md) | One page per application. |
| [Shared file helpers](shared.md) | The `.inp` and Modelica readers' common parts. |
