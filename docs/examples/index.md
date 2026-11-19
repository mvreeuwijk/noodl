# Examples

Six complete worked examples. Each states a goal, runs from a copy-paste block of code, and
shows the result. The code on every page runs as written (the EPANET and CONTAM files it reads
are included in the repository), and each page takes a few seconds on a laptop. Start with the one closest to your field; the first two are the shortest.

<div class="grid cards" markdown>

-   **[Calibrating demands on an EPANET network](calibrate-demands.md)**

    Fit three unknown demands to pressure-logger data with L-BFGS. The gradient of the misfit
    comes from one backward pass through the network solve.

    *Water distribution, gradients*

-   **[A day of pumping on EPANET Net1](water-day.md)**

    A 24 h extended-period run with tank-level pump controls, repeated at three demand levels.

    *Water distribution, time stepping*

-   **[Natural ventilation of two rooms](natural-ventilation.md)**

    A heated and an unheated room joined by a doorway, under a daily outdoor temperature cycle.
    Airflow and temperature are solved together at every step.

    *Building airflow and heat, coupled physics*

-   **[A release in a street network, for every wind direction](street-release.md)**

    A pollutant source in a 12-street grid. Twenty-four wind directions run as one batch.

    *Street air quality, batches*

-   **[Hydrogen sulfide in a sewer under a load surge](sewer-sulfide.md)**

    Hydraulics, air, dissolved sulfide and headspace gas on one sewer tree.

    *Sewers, four layers on one network*

-   **[Street air drawn into a building](street-building.md)**

    A street network and a CONTAM building coupled by exchanging the street concentration
    and the building's infiltration.

    *Coupling two models*

</div>

The figures are produced by `scripts/make_example_figures.py`, which repeats each page's
computation and plots it. To learn the calls the examples use, read the
[Get started](../get-started/index.md) pages; to write your own, start from
[Extending noodl](../project/extending/index.md).
