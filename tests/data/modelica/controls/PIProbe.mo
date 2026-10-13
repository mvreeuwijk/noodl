model PIProbe "LimPID PI with anti-windup on a first-order heated room (probe)"
  Modelica.Blocks.Continuous.LimPID con(
    controllerType=Modelica.Blocks.Types.SimpleController.PI, k=5, Ti=60, Td=10,
    yMax=1, yMin=-1, Ni=0.9, xi_start=0.1);
  Modelica.Blocks.Sources.Constant TSet(k=293.15);
  Real T(start=292.0, fixed=true);
  parameter Real C = 75000;
  parameter Real G = 100;
equation
  connect(TSet.y, con.u_s);
  con.u_m = T;
  C*der(T) = 3000*con.y - G*(T - 273.15);
end PIProbe;
