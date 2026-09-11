# MA-IRS Coverage Enhancement

Independent Python implementation and simulation study based on:

> Y. Gao, Q. Wu, W. Mei, G. Chen, W. Chen, and Z. Zheng,  
> "Integrating Movable Antennas and Intelligent Reflecting Surfaces for Coverage Enhancement,"  
> *IEEE Transactions on Wireless Communications*, 2026.

This repository contains an independently developed Python implementation of the movable antenna and intelligent reflecting surface (MA-IRS) based coverage enhancement system studied in the reference paper.

The implementation is intended for research, learning, simulation, and reproducibility purposes. It is **not an official implementation released by the paper authors**.

The original paper and its contents remain the property of their respective copyright holders.

---

## Overview

The implementation studies coverage enhancement using movable antennas (MAs) and intelligent reflecting surfaces (IRSs).

The simulation framework includes:

- Rician fading channel modeling
- Movable antenna positioning
- IRS phase-shift optimization
- Alternating Optimization (AO)
- Majorization-Minimization (MM) / surrogate-based updates
- Monte Carlo validation
- Analytical SNR evaluation
- Coverage-oriented simulation over multiple target areas
- Numerical studies corresponding to the main simulation figures in the reference paper

The implementation reproduces the main qualitative trends reported in the paper under the documented simulation parameters and assumptions.

---

## Implemented Schemes

The following five schemes are implemented:

1. **Area-adaptive MA-IRS**
2. **Area-adaptive MA-staIRS**
3. **Shared MA-staIRS**
4. **FPA-(area-adaptive IRS)**
5. **FPA-staIRS**

These schemes are evaluated under different system configurations and target-area settings.

---

## Optimization Framework

The main optimization framework uses **Alternating Optimization (AO)** with surrogate/MM-based updates for the IRS phase shifts and movable antenna positions.

The implementation alternates between:

1. Optimizing the IRS phase shifts
2. Optimizing the movable antenna positions
3. Updating the system configuration
4. Evaluating the resulting SNR

The implementation follows the optimization structure described in the reference paper, including surrogate-based updates associated with the corresponding equations and Algorithm 1.

### Important Note

The final implementation does **not** use Block Coordinate Descent (BCD) as the main optimization method.

Instead, it uses:

> **Alternating Optimization (AO) with MM/surrogate-based updates.**

---

## System Parameters

The main simulation parameters follow the numerical evaluation in Section IV of the reference paper.

| Parameter | Value |
|---|---:|
| Carrier frequency | 3 GHz |
| Wavelength | 0.1 m |
| Reference path-loss constant | $(\lambda/4\pi)^2$ |
| IRS path-loss exponent | 2.2 |
| Direct-link path-loss exponent | 3.5 |
| Rician factor | 3 dB |
| Transmit power | 40 dBm |
| Noise power | -90 dBm |
| Number of BS antennas | 4 |
| Minimum MA spacing | $\lambda/2$ |
| IRS element spacing | $\lambda/2$ |
| Number of target areas | 3 |
| Target-area sampling interval | 1 m |

---

## Target Areas

The target areas are modeled in the horizontal plane with:

- $x \in [50,70]$ m
- $y \in [-40,40]$ m
- $z = 0$

Each target area has dimensions:

- **5 m × 5 m**

The target areas are sampled at 1 m intervals for the coverage/SNR evaluation.

---

## IRS Locations

The implementation uses the IRS reference locations corresponding to the configurations considered in the paper:

```text
[5, 0, 12]
[0, 12, 5]
[0, -12, 5]
[10, 25, 5]
[10, -25, 5]
