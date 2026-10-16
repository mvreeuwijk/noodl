"""What the two Modelica readers share.

`noodl.apps.building_physics.modelica` reads Modelica Buildings Library airflow models
(`noodl-modelica/1`); `noodl.apps.electrical.modelica` reads circuits (`noodl-modelica-
electrical/1`). Both read a JSON document an exporter script wrote from OpenModelica's
evaluated instance tree, never Modelica source. This package holds the pipeline they share:
the error they raise, the signal record and the evaluation of Modelica Standard Library
signal blocks (`signals`), the resolution of which signal drives which input (`inputs`),
and the time grid a model is stepped on (`timeline`). It imports no application.
"""

from noodl.modelica.errors import ModelicaImportError

__all__ = ["ModelicaImportError"]
