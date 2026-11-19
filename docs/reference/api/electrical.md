# Electrical circuits

Circuits and power flow. Application guide: [Electrical circuits](../../applications/electrical/index.md).

::: noodl.apps.electrical.analysis
    options:
      members:
        - operating_point
        - initial_state
        - transient
        - ac_sweep
        - device_values

::: noodl.apps.electrical.circuit
    options:
      members:
        - Capacitor
        - Circuit
        - Conductor
        - ControlledSource
        - CurrentSource
        - Diode
        - IdealTransformer
        - Inductor
        - Resistor
        - Switch
        - Transformer
        - VoltageSource
        - build_model

::: noodl.apps.electrical.converters
    options:
      members:
        - reg_root
        - splice
        - ImpedanceTransformer
        - PowerConverter

::: noodl.apps.electrical.laws
    options:
      members:
        - PrescribedVoltage
        - PrescribedCurrent
        - Resistance
        - Capacitance
        - Inductance
        - JunctionDiode
        - SwitchedConductance
        - TransformerRatio
        - MutualInductance
        - DependentSource

::: noodl.apps.electrical.loads
    options:
      members:
        - Impedance
        - ConstantPowerLoad

::: noodl.apps.electrical.modelica
    options:
      members:
        - read_modelica_circuit
        - simulate_circuit
        - CircuitNames
        - Column

::: noodl.apps.electrical.modelica.assemble
    options:
      members:
        - CircuitNames
        - Column
        - build

::: noodl.apps.electrical.modelica.run
    options:
      members:
        - simulate_circuit

::: noodl.apps.electrical.power.build
    options:
      members:
        - PowerFlowIndex
        - PowerFlowSolution
        - check_power_network
        - build_power_flow
        - solve_power_flow
        - power_flow_solution

::: noodl.apps.electrical.power.cases
    options:
      members:
        - MatpowerCaseUnavailableError
        - MatpowerCaseDigestError
        - matpower_case

::: noodl.apps.electrical.power.dc
    options:
      members:
        - DcPowerFlowSolution
        - dc_power_flow
        - dc_power_flow_solution

::: noodl.apps.electrical.power.laws
    options:
      members:
        - PiBranch
        - Shunt
        - VoltageControlledGenerator

::: noodl.apps.electrical.power.matpower
    options:
      members:
        - read_matpower

::: noodl.apps.electrical.power.network
    options:
      members:
        - Bus
        - Branch
        - Generator
        - Load
        - FixedShunt
        - PowerNetwork

::: noodl.apps.electrical.thermal
    options:
      members:
        - HeatedEdge
        - JouleHeating

::: noodl.apps.electrical.waveforms
    options:
      members:
        - Constant
        - Step
        - Ramp
        - Sine
        - Pulse
        - Signal
