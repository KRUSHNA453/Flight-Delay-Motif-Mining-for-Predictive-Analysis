# ✈️ Flight Delay Motif Mining for Predictive Analysis

This research project explores **Temporal Delay Motifs** and **Causal Discovery** to enhance flight delay prediction using a **Spatio-Temporal Graph Neural Network (STGNN)**. 

Moving beyond simple delay classification, this pipeline models how delays *propagate* through the US airport network by identifying unique 3-node propagation chains.

## 📊 Key Results

- **MAE Improvement**: **23.24%** (Reduction from 14.2 to 10.9 min)
- **RMSE Improvement**: **22.17%** (Reduction from 22.1 to 17.2 min)
- **MAPE Improvement**: **33.33%** (Reduction from 0.18 to 0.12)
- **Causal Discovery Speedup**: **45.2%** (Computational acceleration via motif-filtering)

## 🔬 Pipeline Overview

1.  **Dataset Integration**: Integration of 2023 US flight data with weather metrics and airport geolocations.
2.  **Temporal Graph Construction**: Building dynamic graphs with 6-hour snapshots.
3.  **DM-Miner**: Mining 233 unique 3-node temporal motifs (delay propagation chains).
4.  **PCMCI Causal Discovery**: Using motif-driven priors to accelerate causal validation by 45%.
5.  **STGNN Prediction**: A Spatio-Temporal GNN architecture for high-accuracy delay forecasting.

## 🛠️ Tech Stack

- **Graph Learning**: PyTorch Geometric
- **Causal Analysis**: Tigramite / PCMCI
- **Dashboard**: Streamlit
- **Visualisation**: Plotly, PyDeck, PyVis, Folium
- **Data Science**: Pandas, NumPy, NetworkX, Scikit-learn

## 🚀 How to Run

### Interactive Dashboard
To explore the motifs and run simulated predictions:
```powershell
pip install -r streamlit_app/requirements.txt
streamlit run streamlit_app/app.py
```

### Research Notebook
To run the full end-to-end analysis, motif mining, and model training:
1.  Open `flight_delay_motifs.ipynb` in VS Code or Jupyter.
2.  Ensure you have a CUDA-enabled environment (optimized for CUDA 12.4).
3.  Execute all cells to reproduce the results.

## 📂 Datasets
The study utilizes the [2023 US Civil Flights Delay, Meteo, and Aircraft](https://www.kaggle.com/datasets/bordanova/2023-us-civil-flights-delay-meteo-and-aircraft) dataset.

---
*Built with ❤️ as part of the 2024-25 Research Pipeline.*
