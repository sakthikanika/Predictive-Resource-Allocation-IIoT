# Predictive Resource Allocation for Mobility-Aware Task Offloading and Migration in Edge Environments

This repository contains the final version of the research paper submitted for conference consideration. It presents an improved framework for computation offloading and task migration in Industrial Internet of Things (IIoT) environments, with an emphasis on predictive mobility and resource-aware decision-making.

---

## Table of Contents

- [Abstract](#abstract)
- [Research Objectives](#research-objectives)
- [System Overview](#system-overview)
- [Key Contributions](#key-contributions)
- [Performance Evaluation](#performance-evaluation)
- [File Description](#file-description)
- [Citation](#citation)
- [Authors](#authors)
- [Contact](#contact)

---

## Abstract

Mobility-enabled devices in IIoT face computational limitations due to constrained battery life and onboard processing capabilities. To alleviate this, edge computing supports the offloading of intensive tasks to edge servers. However, existing offloading frameworks often lack adaptability in dynamic mobility environments and suffer from inaccurate trajectory prediction and inefficient resource usage.

This work introduces a Newton-enhanced MCOTM (Mobility-aware Computation Offloading and Task Migration) framework that leverages Newton Forward Interpolation for precise mobility prediction, LSTM networks for system resource forecasting, and Deep Deterministic Policy Gradient (DDPG) for adaptive offloading and migration decisions. Experimental results demonstrate significant improvements in turnaround time, energy usage, and task migration rates when compared with conventional methods.

---

## Research Objectives

- Enhance trajectory prediction in IIoT using Newton Forward Interpolation.
- Accurately forecast system resource availability using LSTM models.
- Optimize decision-making for offloading and migration using reinforcement learning (DDPG).
- Improve overall efficiency and responsiveness in edge computing environments.

---

## System Overview

The Newton-enhanced MCOTM framework consists of three integrated layers:

1. **Trajectory Prediction Layer**:  
   Applies Newton’s Forward Interpolation to estimate future device positions, enhancing prediction stability and reducing computational cost.

2. **Resource Forecasting Layer**:  
   Utilizes LSTM neural networks to predict future availability of resources on both user and edge nodes, enabling proactive planning.

3. **Decision Optimization Layer**:  
   Employs DDPG, a deep reinforcement learning algorithm, to determine optimal task offloading and migration strategies based on current and predicted system states.

---

## Key Contributions

- Mathematical improvement by replacing Lagrange Interpolation with Newton’s method.
- Integration of LSTM for accurate and dynamic resource forecasting.
- Use of DDPG for real-time optimization of offloading and migration decisions.
- Performance improvements over the original MCOTM framework:
  - 7.76% reduction in turnaround time
  - 6.96% decrease in energy consumption
  - 4.55% reduction in task migration rate

---

## Performance Evaluation

| Metric              | Lagrange-MCOTM | Newton-MCOTM | Improvement |
|---------------------|----------------|--------------|-------------|
| Turnaround Time     | 142 ms         | 131 ms       | ↓ 7.76%     |
| Energy Consumption  | 0.81 J         | 0.75 J       | ↓ 6.96%     |
| Migration Rate      | 50.2%          | 47.9%        | ↓ 4.55%     |

These improvements indicate better task scheduling and energy efficiency, which are critical for sustainable and responsive IIoT systems.

---

## File Description

- `Predictive Resource Allocation for Mobility-Aware
Task Offloading and Migration in Edge
Environments.pdf` – Final version of the paper in IEEE conference format.

---

## Citation

If you wish to cite this work in your research, please use the following BibTeX entry:

```bibtex
@inproceedings{sakthikanika2025mcotm,
  title={Predictive Resource Allocation for Mobility-Aware Task Offloading and Migration in Edge Environments},
  author={Atchaya R and Sakthikanika V and Rakavi Dharshini K and Kannan K and Ezhilarasie R},
  booktitle={Proceedings of [Conference Name]},
  year={2025},
  institution={SASTRA Deemed University}
}
```

---

## Authors

- **Atchaya R**  
- **Sakthikanika V**  
- **Rakavi Dharshini K**  
- **Kannan K**  
- **Ezhilarasie R**  
School of Computing, SASTRA Deemed University, Thanjavur, India

---

## Contact

**Sakthikanika V**  
Email: 126003229@sastra.ac.in  
Institution: SASTRA Deemed University, Tamil Nadu, India
