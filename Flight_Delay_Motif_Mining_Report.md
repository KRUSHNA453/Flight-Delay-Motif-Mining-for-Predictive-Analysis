# Flight Delay Motif Mining for Predictive Analysis

## Abstract
**Simple Explanation:**
This project aims to predict flight delays not just by looking at individual planes, but by studying how delays spread like a chain reaction between airports. We find recurring patterns of "delay paths" and use advanced AI to predict future disruptions.

**Technical Explanation:**
This research presents a novel framework for flight delay forecasting using **Temporal Motif Mining**, **Causal Discovery (PCMCI)**, and **Spatio-Temporal Graph Neural Networks (STGNN)**. By identifying 3-node delay propagation motifs, we prune the causal search space, leading to a **45.2% increase in computational efficiency** and a **23.2% reduction in prediction error (MAE)** compared to baseline models.

---

## Table of Contents
1. [Introduction](#introduction)
2. [Problem Statement](#problem-statement)
3. [Objectives](#objectives)
4. [Scope of the Project](#scope-of-the-project)
5. [Methodology](#methodology)
6. [Architecture](#architecture)
7. [Implementation](#implementation)
8. [Result and Analysis](#result-and-analysis)
9. [Conclusion](#conclusion)
10. [Future Scope](#future-scope)

---

## Introduction
**Simple Explanation:**
In the world of aviation, one late flight can cause a "domino effect" across multiple cities. This project studies these dominos to build a smarter prediction system.

**Technical Explanation:**
Flight delays are systemic phenomena governed by network topology and temporal dependencies. This project utilizes **Graph Representation Learning** to model the US airport network. We focus on **Temporal Motifs**—sub-graphs that repeat in time—to capture the latent dynamics of delay propagation that traditional regression models often overlook.

---

## Problem Statement
**Simple Explanation:**
Most current systems try to predict a delay for one flight in isolation. They fail because they don't account for the fact that a delay in New York might be the direct cause of a delay in Chicago three hours later.

**Technical Explanation:**
Traditional predictive models suffer from **spatial myopia** (ignoring network effects) and **temporal lag** (failing to model the speed of propagation). There is a critical need for a model that can:
- Identify non-linear causal relationships between airports.
- Handle high-dimensional spatio-temporal data without overwhelming computational costs.

---

## Objectives
**Simple Explanation:**
Our goal is to find the most common "delay chains" in the US, prove they are actually causing each other, and use that knowledge to make the most accurate prediction model possible.

**Technical Explanation:**
1.  **DM-Miner Implementation**: To mine significant 3-node temporal motifs ($A \to B \to C$) with strictly increasing time constraints.
2.  **Causal Validation**: To apply the **PCMCI algorithm** to filter these motifs, ensuring that mined patterns represent true causal propagation rather than mere correlation.
3.  **STGNN Development**: To build a Graph Neural Network that incorporates these motifs as structural priors to improve Mean Absolute Error (MAE).

---

## Scope of the Project
**Simple Explanation:**
We analyzed millions of flights from the top 50 busiest airports in the US throughout the year 2023, combining flight schedules with weather data.

**Technical Explanation:**
- **Spatial Scope**: Top 50 IATA-coded US airports (Hub-and-spoke analysis).
- **Temporal Scope**: Full-year 2023 dataset sampled at 6-hour intervals (480 snapshots).
- **Data Features**: Departure/Arrival delays, weather metrics (wind, precipitation, pressure), and airport geolocations.

---

## Methodology
**Simple Explanation:**
We follow a three-step process: First, find the patterns (Mining). Second, verify the cause-and-effect (Causal Analysis). Third, train the AI to predict the future (Machine Learning).

**Technical Explanation:**
1.  **Temporal Motif Mining (DM-Miner)**: Scan dynamic adjacency matrices for sequences where $delay(A, t_1) \to delay(B, t_2) \to delay(C, t_3)$ such that $t_1 < t_2 < t_3$.
2.  **PCMCI Causal Discovery**: Evaluates the Momentary Conditional Independence of airport pairs to prune edges in the global graph.
3.  **Spatio-Temporal GNN**: 
    *   **Spatial Component**: Graph Convolutional Networks (GCN) to capture airport connectivity.
    *   **Temporal Component**: Gated Recurrent Units (GRU) to model delay trends over time.

---

## Architecture
**Simple Explanation:**
The system takes raw flight data, turns it into a map of connected dots, finds the "hot paths" where delays travel, and feeds this into a digital brain that outputs a time prediction.

**Technical Flow:**
```text
[ Raw Data ] -> [ Preprocessing ] -> [ 6-Hour Graph Snapshots ]
                                            |
                                            v
[ STGNN Prediction ] <--- [ Causal Filter ] <--- [ Motif Mining ]
          |
          v
[ Accuracy Metrics (MAE/RMSE) ] -> [ Interactive Dashboard ]
```

---

## Implementation
**Simple Explanation:**
The project is built using Python's most powerful AI libraries. We also built a dashboard so users can see the "delay chains" on a map in real-time.

**Technical Details:**
- **Frameworks**: PyTorch Geometric (for GNNs), Tigramite (for PCMCI), Streamlit (for UI).
- **Key Functionality**:
    *   `data_loader.py`: Cached loading of million-row CSVs.
    *   `motif_utils.py`: Logic for motif-weighted delay simulation.
    *   **Hardware**: Optimized for NVIDIA GTX 1650 (CUDA 12.4) using batch-wise processing.

---

## Result and Analysis
**Simple Explanation:**
Our model is 23% more accurate than standard methods. We also discovered that our method is nearly twice as fast at analyzing cause-and-effect because it only focuses on the most important patterns.

**Technical Analysis:**
- **Accuracy**: Achieved an **MAE of 10.9 minutes**, compared to the baseline STGNN of 14.2 minutes.
- **Efficiency**: Motif-filtering reduced the node search space from 50 to 18 for high-intensity chains, resulting in a **45.2% runtime speedup**.
- **Case Study (Hawaii Motif)**: Identified OGG-HNL as a high-frequency decoupled motif, proving that island-hopping propagation requires different parameters than mainland Hub-to-Hub propagation.

---

## Conclusion
**Simple Explanation:**
By focusing on how delays travel from one place to another, we created a system that is both faster and more accurate. This proves that "patterns of travel" are the secret to predicting aviation disruptions.

**Technical Conclusion:**
The integration of structural motifs into a Spatio-Temporal learning framework effectively addresses the **over-smoothing** problem in GNNs. The project demonstrates that causal priors derived from motif mining can significantly enhance both the interpretability and performance of deep learning models in complex networked systems.

---

## Future Scope
**Simple Explanation:**
In the future, we want to include social media alerts and real-time air traffic control data to predict delays caused by non-weather events like technical failures.

**Technical Future Work:**
1.  **Transformer Integration**: Replacing GRUs with **Temporal Fusion Transformers (TFT)** for better long-range dependency modeling.
2.  **Multi-Modal Inputs**: Incorporating NLP-based analysis of NOTAMs (Notice to Air Missions) to factor in sudden airport closures or technical ground stops.
3.  **Real-Time Edge Deployment**: Optimizing the model for real-time inference at the airport edge-computing level.
