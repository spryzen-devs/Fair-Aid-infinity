# FairAid-Infinity ⚖️

**Intelligent Aid Allocation System** designed for hackathons to demonstrate fair, sector-aware resource distribution using AI-driven logic.

## 🚀 How to Run (Setup Guide)

Follow these steps to run the application on any device (Windows, Mac, or Linux).

### Prerequisites
- **Python 3.10+** installed. ([Download Python](https://www.python.org/downloads/))
- **Git** installed. ([Download Git](https://git-scm.com/downloads))

### 1. Clone the Repository
Open your terminal (Command Prompt, PowerShell, or Terminal) and run:
```bash
git clone https://github.com/spryzen-devs/Fair-Aid-infinity.git
cd Fair-Aid-infinity
```

### 2. Create a Virtual Environment (Optional but Recommended)
**Windows:**
```bash
python -m venv venv
.\venv\Scripts\activate
```
**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application
```bash
streamlit run app.py
```
*The app should automatically open in your browser at `http://localhost:8501` (or similar).*

---

## 💡 How It Works (The Winning Flow)

1.  **Select Sector:** Choose Education, Health, Food Security, or Disaster Relief.
2.  **Upload Data:** Upload one or more CSV files.
3.  **Auto-Map & Proxy:** The system uses AI to:
    *   **Identify** columns automatically (e.g., finding "Student Count" for Population).
    *   **Generate Proxies** for missing data (e.g., estimating Dropout Rate from Literacy Rate).
4.  **Analyze & Reallocate:**
    *   View the "Fairness Score" for each region.
    *   Run the **Simulation** to see the "Tax & Distribute" model in action.

## 🛠️ Tech Stack
-   **Frontend:** Streamlit
-   **Data Processing:** Pandas, NumPy
-   **Visualization:** Plotly Express
-   **Logic:** Custom Fairness & Reallocation Engines
