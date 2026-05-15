To use the damping_analysis script, simply edit the `FIELD_KEY` variable to either the Electric Field or the Density Amplitude.

```
FIELD_KEY = "rho_mode_amp"

FIELD_LABELS = {
    "E_amp":        "|E|_max",
    "rho_mode_amp": "|δρ_k| (density mode)",
}
```

For dependencies, the analytical damping rate is calculated by an external library plasmadisp, available via - [text](https://github.com/ergodicio/plasmadisp.git) 

