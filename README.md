# DarkELF

DarkELF is a python package capable of calculating interaction rates of light dark matter in dielectric materials, including screening effects. The full response of the material is parametrized in the terms of the energy loss function (ELF) of material, which DarkELF converts into differential scattering rates for both direct dark matter electron scattering and through the Migdal effect. In addition, DarkELF can calculate the rate to produce phonons from sub-MeV dark matter scattering via the dark photon mediator, as well as the absorption rate for dark matter comprised of dark photons. The package currently includes precomputed ELFs for Al,A_2O_3, GaAs, GaN, Ge, Si, SiO_2, and ZnS, and allows the user to easily add their own ELF extractions for arbitrary materials.

See arXiv xxxx.xxxx for a description of the implementation

## Authors

Simon Knapen, Jonathan Kozaczuk and Tongyan Lin

## Physics
### ELF

Currently DarkELF contains ELF look-up tables obtained with the [GPAW](https://wiki.fysik.dtu.dk/gpaw/) density functional theory code for Si and Ge, as well as data-driven fits to a Mermin model for the remaining materials. The Lindhard ELF is also included. DarkELF also comes with a number of measured ELFs in the optical limit for energy depositions below the electronic band gap, which is relevant for phonon processes. Additional materials and ELF computations may be added at a later date. When using a particular ELF computation, please refer to the relevant experimental papers and/or GPAW package. These references can be found in arXiv xxxx.xxxx. 

### DM electron scattering

DarkELF can calculates DM-electron scattering rates in dielectric materials by making use of ELF of the material, as described in arXiv [2101.08275](https://arxiv.org/abs/2101.08275). This approach automatically includes screening effects.

### DM - nucleon scattering through the Migdal effect

DarkELF can compute the rate for ionizations through the Migdal effect, as derived in arXiv [2011.09496](https://arxiv.org/abs/2011.09496). Currently we provide calculations in the soft limit, assuming the free ion or impulse approximation. For a number of materials we also include the tabulated form factors for the atomic Migdal effect, as computed by Ibe et. al. (arXiv [1707.07258](https://arxiv.org/abs/1707.07258)).

### DM - phonon scattering

DM - phonon scattering through a light dark photon mediator (arXiv [1712.06598](https://arxiv.org/abs/1712.06598), [1807.10291](https://arxiv.org/abs/1807.10291)) can also be computedd in terms of the ELF.

### DM - Absorption

If the DM is itself a light dark photon, it can be absorbed on phonons (arXiv [1712.06598](https://arxiv.org/abs/1712.06598), [1807.10291](https://arxiv.org/abs/1807.10291)) or electrons (arXiv [1608.02123](https://arxiv.org/abs/1608.02123), [1608.01994](https://arxiv.org/abs/1608.01994)), the rate of which can be extracted directly from the ELF.

## Requirements and Usage

DarkELF requires python 3.6 or higher, equipped with the numpy, scipy, pyyaml and pandas packages. The tutorial notebooks require a jupyter installation, but this is in general not needed for DarkELF itself.

The examples folder contains a tutorial jupyter notebook for each of the processes outlined above. Appendix A of the DarkELF paper contains additional explanations of the various functions and settings that are available to the user. 

## Citation guide

If you use DarkELF, please refer to the corresponding paper arXiv xxxx.xxxx. For the specific processes, please refer to the papers outlined in the sections above.
