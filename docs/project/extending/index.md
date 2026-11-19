# Extending noodl

noodl can be extended in three ways, each without changing the core. They build on one another:
a reader feeds an application, an application is composed from layers and laws, and a law is the
smallest piece.

| You want to | Do this | Page |
|---|---|---|
| Add a new relation between potential difference and flow on an edge (a new valve, a new friction law, a new heat-transfer correlation) | Subclass `ConstitutiveLaw` | [A new constitutive law](law.md) |
| Model a new kind of system from the existing layers and laws | Write a builder that returns `(model, state, drivers)`, with its documentation and tests | [A new application](application.md) |
| Read a model from a file format noodl does not read yet | Write a reader next to the application it feeds | [A new file reader](reader.md) |

Start with the law page: the other two use its example. Everything on these pages goes through
noodl's ordinary public surface (`noodl.elements`, `noodl.topology`, `noodl.layers.kirchhoff`,
`noodl.model`), and the code in the recipe is exercised by the repository's test suite, so it
runs as shown.

**Numerical blocks are not extension points (yet).** A new solver or layer class is a framework
change, not an extension: it needs the same design, testing and review as any other change
under `src/noodl/layers` or `src/noodl/solvers`. [Contributing](../contributing.md) describes
that process, and the [design notes](../design/index.md) record why the existing blocks are
built as they are.
