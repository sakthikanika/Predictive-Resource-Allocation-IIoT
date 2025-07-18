# Predictive Resource Allocation for Mobility-Aware Task Offloading and Migration in Edge Environments

This repository contains two related projects that explore task offloading and migration strategies in Industrial Internet of Things (IIoT) environments using the MCOTM framework:

- **Project 1:** Baseline MCOTM with Lagrange Interpolation  
- **Project 2:** Newton-Enhanced MCOTM with trajectory prediction improvements and deep learning enhancements

Both projects are part of our research work submitted for conference consideration.

---

## Table of Contents

- [Overview](#overview)
- [Project 1: MCOTM Execution](#project-1-mcotm-execution)
  - [Abstract](#abstract-for-project-1)
- [Project 2: Newton-Enhanced MCOTM](#project-2-newton-enhanced-mcotm)
  - [Abstract](#abstract-for-project-2)
- [Performance Evaluation](#performance-evaluation)
- [File Description](#file-description)
- [Citation](#citation)
- [Authors](#authors)
- [Contact](#contact)

---

## Overview

This repository explores two approaches to computation offloading and task migration in IIoT:

- **MCOTM Execution:** Uses Lagrange Interpolation for device trajectory prediction, combined with a DDPG-based offloading agent.
- **Newton-Enhanced MCOTM:** Replaces Lagrange with Newton Forward Interpolation for improved mobility prediction and integrates LSTM-based resource forecasting.

---

## Project 1: MCOTM Execution

### Abstract for Project 1

In Industrial Internet of Things (IIoT) environments, mobile devices such as sensors, robots, and edge nodes are often constrained by limited computational capabilities and energy resources. To meet the increasing demand for real-time processing, computation offloading to edge servers has become a practical solution.

This project implements the baseline MCOTM (Mobility-aware Computation Offloading and Task Migration) framework, which integrates Lagrange Interpolation for trajectory prediction and the Deep Deterministic Policy Gradient (DDPG) algorithm for decision-making. It aims to improve task scheduling by adapting to device mobility and edge resource constraints in real-time.

---

## Project 2: Newton-Enhanced MCOTM

### Abstract for Project 2

Mobility-enabled devices in IIoT face computational limitations due to constrained battery life and onboard processing capabilities. To alleviate this, edge computing supports the offloading of intensive tasks to edge servers. However, existing offloading frameworks often lack adaptability in dynamic mobility environments and suffer from inaccurate trajectory prediction and inefficient resource usage.

This project enhances the original MCOTM framework by replacing Lagrange Interpolation with Newton Forward Interpolation for mobility prediction. It also integrates Long Short-Term Memory (LSTM) networks for system resource forecasting and applies the Deep Deterministic Policy Gradient (DDPG) algorithm for adaptive task migration. This results in improved turnaround time, energy efficiency, and reduced task migration rates.

---

## Performance Evaluation

| Metric              | Lagrange-MCOTM | Newton-MCOTM | Improvement |
|---------------------|----------------|--------------|-------------|
| Turnaround Time     | 142 ms         | 131 ms       | ↓ 7.76%     |
| Energy Consumption  | 0.81 J         | 0.75 J       | ↓ 6.96%     |
| Migration Rate      | 50.2%          | 47.9%        | ↓ 4.55%     |

These results demonstrate the benefit of integrating Newton Interpolation and deep learning components into offloading strategies for IIoT.

---

## File Description

- `mcotm-execution/` – Implementation of original MCOTM using Lagrange Interpolation
- `newton-enhanced-mcotm/` – Enhanced MCOTM using Newton Interpolation + LSTM
- `figures/` – Diagrams and performance graphs
- `LICENSE` – Project license
- `README.md` – This file

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
