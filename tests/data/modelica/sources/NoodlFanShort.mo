within ;
package NoodlFanShort
  "Airflow reference models for FlowControlled_dp fans and zero-resistance connections"
  extends Modelica.Icons.Package;

  model FanOrifice
    "A constant-head fan drives an orifice between two equal pressures"
    extends Modelica.Icons.Example;
    package Medium = Buildings.Media.Specialized.Air.PerfectGas;
    Buildings.Fluid.Sources.Boundary_pT bouA(redeclare package Medium = Medium,
      p=101325, T=293.15, nPorts=1) "Pressure boundary";
    Buildings.Fluid.Sources.Boundary_pT bouB(redeclare package Medium = Medium,
      p=101325, T=293.15, nPorts=1) "Pressure boundary";
    Buildings.Fluid.Movers.FlowControlled_dp fan(redeclare package Medium = Medium,
      m_flow_nominal=0.05, inputType=Buildings.Fluid.Types.InputType.Constant,
      constantHead=5, use_riseTime=false, addPowerToMedium=false,
      energyDynamics=Modelica.Fluid.Types.Dynamics.SteadyState) "Fan";
    Buildings.Airflow.Multizone.Orifice ori(redeclare package Medium = Medium,
      A=0.01, CD=0.6) "Orifice";
    Buildings.Fluid.Sensors.MassFlowRate senFan(redeclare package Medium = Medium)
      "Flow leaving the fan";
    Buildings.Fluid.Sensors.MassFlowRate senOri(redeclare package Medium = Medium)
      "Flow leaving the orifice";
  equation
    connect(bouA.ports[1], fan.port_a);
    connect(fan.port_b, senFan.port_a);
    connect(senFan.port_b, ori.port_a);
    connect(ori.port_b, senOri.port_a);
    connect(senOri.port_b, bouB.ports[1]);
    annotation (experiment(StopTime=60, Interval=1, Tolerance=1e-6));
  end FanOrifice;

  model FanRamp
    "A fan whose pressure rise follows a ramp (dp_in) drives an orifice"
    extends Modelica.Icons.Example;
    package Medium = Buildings.Media.Specialized.Air.PerfectGas;
    Buildings.Fluid.Sources.Boundary_pT bouA(redeclare package Medium = Medium,
      p=101325, T=293.15, nPorts=1) "Pressure boundary";
    Buildings.Fluid.Sources.Boundary_pT bouB(redeclare package Medium = Medium,
      p=101325, T=293.15, nPorts=1) "Pressure boundary";
    Buildings.Fluid.Movers.FlowControlled_dp fan(redeclare package Medium = Medium,
      m_flow_nominal=0.05, inputType=Buildings.Fluid.Types.InputType.Continuous,
      use_riseTime=false, addPowerToMedium=false,
      energyDynamics=Modelica.Fluid.Types.Dynamics.SteadyState) "Fan";
    Buildings.Airflow.Multizone.Orifice ori(redeclare package Medium = Medium,
      A=0.01, CD=0.6) "Orifice";
    Buildings.Fluid.Sensors.MassFlowRate senFan(redeclare package Medium = Medium)
      "Flow leaving the fan";
    Buildings.Fluid.Sensors.MassFlowRate senOri(redeclare package Medium = Medium)
      "Flow leaving the orifice";
    Modelica.Blocks.Sources.Ramp ramp(height=20, duration=20, offset=0, startTime=10)
      "Pressure rise";
  equation
    connect(ramp.y, fan.dp_in);
    connect(bouA.ports[1], fan.port_a);
    connect(fan.port_b, senFan.port_a);
    connect(senFan.port_b, ori.port_a);
    connect(ori.port_b, senOri.port_a);
    connect(senOri.port_b, bouB.ports[1]);
    annotation (experiment(StopTime=60, Interval=1, Tolerance=1e-6));
  end FanRamp;

  model FanAdverse
    "A constant-head fan against a ramping back pressure: the flow reverses"
    extends Modelica.Icons.Example;
    package Medium = Buildings.Media.Specialized.Air.PerfectGas;
    Buildings.Fluid.Sources.Boundary_pT bouA(redeclare package Medium = Medium,
      p=101325, T=293.15, nPorts=1) "Pressure boundary";
    Buildings.Fluid.Sources.Boundary_pT bouB(redeclare package Medium = Medium,
      use_p_in=true, T=293.15, nPorts=1) "Pressure boundary";
    Buildings.Fluid.Movers.FlowControlled_dp fan(redeclare package Medium = Medium,
      m_flow_nominal=0.05, inputType=Buildings.Fluid.Types.InputType.Constant,
      constantHead=10, use_riseTime=false, addPowerToMedium=false,
      energyDynamics=Modelica.Fluid.Types.Dynamics.SteadyState) "Fan";
    Buildings.Airflow.Multizone.Orifice ori(redeclare package Medium = Medium,
      A=0.01, CD=0.6) "Orifice";
    Buildings.Fluid.Sensors.MassFlowRate senOri(redeclare package Medium = Medium)
      "Flow leaving the orifice";
    Modelica.Blocks.Sources.Ramp ramp(height=30, duration=60, offset=101325 - 10)
      "Back pressure, from 10 Pa below to 20 Pa above the inlet";
  equation
    connect(ramp.y, bouB.p_in);
    connect(bouA.ports[1], fan.port_a);
    connect(fan.port_b, ori.port_a);
    connect(ori.port_b, senOri.port_a);
    connect(senOri.port_b, bouB.ports[1]);
    annotation (experiment(StopTime=60, Interval=1, Tolerance=1e-6));
  end FanAdverse;

  model ShortConnections
    "LosslessPipe and PressureDrop at dp_nominal = 0 in series with an orifice"
    extends Modelica.Icons.Example;
    package Medium = Buildings.Media.Specialized.Air.PerfectGas;
    Buildings.Fluid.Sources.Boundary_pT bouA(redeclare package Medium = Medium,
      use_p_in=true, T=293.15, nPorts=1) "Pressure boundary";
    Buildings.Fluid.Sources.Boundary_pT bouB(redeclare package Medium = Medium,
      p=101325, T=293.15, nPorts=1) "Pressure boundary";
    Buildings.Fluid.FixedResistances.LosslessPipe pip(redeclare package Medium = Medium,
      m_flow_nominal=0.05) "Zero-resistance pipe";
    Buildings.Fluid.FixedResistances.PressureDrop res(redeclare package Medium = Medium,
      m_flow_nominal=0.05, dp_nominal=0) "Zero pressure drop";
    Buildings.Airflow.Multizone.Orifice ori(redeclare package Medium = Medium,
      A=0.01, CD=0.6) "Orifice";
    Buildings.Fluid.Sensors.MassFlowRate senOri(redeclare package Medium = Medium)
      "Flow leaving the orifice";
    Modelica.Blocks.Sources.Ramp ramp(height=100, duration=60, offset=101325 - 50)
      "Pressure at A, from 50 Pa below to 50 Pa above B";
  equation
    connect(ramp.y, bouA.p_in);
    connect(bouA.ports[1], pip.port_a);
    connect(pip.port_b, res.port_a);
    connect(res.port_b, ori.port_a);
    connect(ori.port_b, senOri.port_a);
    connect(senOri.port_b, bouB.ports[1]);
    annotation (experiment(StopTime=60, Interval=1, Tolerance=1e-6));
  end ShortConnections;

  model FanShortParallel
    "A fan through a zero-resistance pipe in parallel with a leak, one boundary pair"
    extends Modelica.Icons.Example;
    package Medium = Buildings.Media.Specialized.Air.PerfectGas;
    Buildings.Fluid.Sources.Boundary_pT bouA(redeclare package Medium = Medium,
      p=101325, T=293.15, nPorts=2) "Pressure boundary";
    Buildings.Fluid.Sources.Boundary_pT bouB(redeclare package Medium = Medium,
      use_p_in=true, T=293.15, nPorts=2) "Pressure boundary";
    Buildings.Fluid.Movers.FlowControlled_dp fan(redeclare package Medium = Medium,
      m_flow_nominal=0.05, inputType=Buildings.Fluid.Types.InputType.Constant,
      constantHead=8, use_riseTime=false, addPowerToMedium=false,
      energyDynamics=Modelica.Fluid.Types.Dynamics.SteadyState) "Fan";
    Buildings.Fluid.FixedResistances.LosslessPipe pip(redeclare package Medium = Medium,
      m_flow_nominal=0.05) "Zero-resistance pipe after the fan";
    Buildings.Airflow.Multizone.Orifice oriFan(redeclare package Medium = Medium,
      A=0.01, CD=0.6) "Orifice in series with the fan";
    Buildings.Airflow.Multizone.Orifice oriLeak(redeclare package Medium = Medium,
      A=0.004, CD=0.6) "Leak between the same boundaries";
    Buildings.Fluid.Sensors.MassFlowRate senFan(redeclare package Medium = Medium)
      "Flow leaving the fan branch";
    Buildings.Fluid.Sensors.MassFlowRate senLeak(redeclare package Medium = Medium)
      "Flow leaving the leak";
    Modelica.Blocks.Sources.Ramp ramp(height=16, duration=60, offset=101325 - 4)
      "Back pressure";
  equation
    connect(ramp.y, bouB.p_in);
    connect(bouA.ports[1], fan.port_a);
    connect(fan.port_b, pip.port_a);
    connect(pip.port_b, oriFan.port_a);
    connect(oriFan.port_b, senFan.port_a);
    connect(senFan.port_b, bouB.ports[1]);
    connect(bouA.ports[2], oriLeak.port_a);
    connect(oriLeak.port_b, senLeak.port_a);
    connect(senLeak.port_b, bouB.ports[2]);
    annotation (experiment(StopTime=60, Interval=1, Tolerance=1e-6));
  end FanShortParallel;
end NoodlFanShort;
