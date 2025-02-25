# DropPyTension

## Description
A small project to read drop images, extract drop parameters, and calculate interfacial tension without complex optimization (which makes its use and installation quite straightforwards). As a matter of fact, I started this project because I wasn't able to use the other available projects I found. The tradeoff for simplicity is an uncertainty of about 1-3 mN/m in the measured interfacial tention depending on the measurement conditions.

<div align="center">
  <img src="https://raw.githubusercontent.com/KevinBallu/DropPyTension/main/example/final_analysis.png" width="400">
</div>


The code uses the approaches proposed by:  
Andreas, J. M.; Hauser, E. A.; Tucker, W. B. Boundary Tension by Pendant Drops. J. Phys. Chem. 1938, 42 (8), 1001–1019. https://doi.org/10.1021/j100903a002.

In which the 1/H values are calculated from the expressions given in:
Drelich J, Fang C, White CL (2002) Measurement of interfacial tension in fluid-fluid systems. Encycl Surf Colloid Sci 3:3158–3163


## Licence
GNU General Public License v3.0

## Installation

### Clone the Repository
```bash
git clone https://github.com/KevinBallu/DropPyTension.git
```
## Citation

If you use this project in your work, please cite the following publication:

**Author(s)**. (Year). *Title of the Paper*. *Journal Name*, Volume(Issue), Page Range. DOI/Publisher.
