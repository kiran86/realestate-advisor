# RealEstate Advisor 🚀

A Python-based data exploration and visualization project focused on Indian housing prices. This repository provides tools and a Streamlit application to analyze real estate trends, perform exploratory data analysis, and offer insights for potential buyers or investors.

---

## 🗂️ Repository Structure

```
README.md

data/
    gadm41_IND_2.json            # GeoJSON for Indian administrative boundaries (level 2)
    india_housing_prices.csv     # Dataset of housing prices and features
notebooks/
    eda.ipynb                    # Exploratory data analysis notebook
streamlit_app/
    app.py                       # Streamlit web application
```

---

## 🔍 Features

- Interactive exploratory data analysis using Jupyter Notebook
- Visualizations of price distributions, trends, and geographic patterns
- Streamlit-based UI for non‑technical users to query and visualize data
- Geographic mapping of prices using Indian boundary data

---

## ⚙️ Getting Started

### Prerequisites

Ensure you have Python 3.8+ installed.

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/kiran86/realestate-advisor.git
   cd realestate-advisor
   ```

2. (Optional) Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate            # on Linux/macOS
   venv\Scripts\activate             # on Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

> **Note:** If a `requirements.txt` file is not present, install packages manually such as `pandas`, `geopandas`, `streamlit`, `matplotlib`, etc.

---

## 🧪 Usage

### Exploratory Notebook

Open `notebooks/eda.ipynb` with Jupyter:
```bash
jupyter notebook notebooks/eda.ipynb
```

### Streamlit Application

From the project root, run:
```bash
streamlit run streamlit_app/app.py
```

The app provides a web interface to filter and visualize housing price data by region, property type, and other features.

---

## 📁 Data Sources

- `india_housing_prices.csv`: Publicly available dataset (ensure proper attribution if reused)
- `gadm41_IND_2.json`: GADM level‑2 boundaries for India used for mapping

> **Tip:** Replace/augment these files with updated data as needed. Consider adding a data ingestion script for automation.

---

## 🛠️ Development

- Add new analyses or visualizations in the notebook
- Extend the Streamlit app with additional filters or charts
- Validate models or integrate a prediction component (future work)

For code contributions, follow standard GitHub workflows with branches and pull requests.

---

## **Deploying to Streamlit Cloud**

- **Repository:** Push your repository to GitHub (the `main` branch is fine).
- **Entry point:** When you create the app on Streamlit Cloud, set the Python file to `streamlit_app/app.py`.
- **Dependencies:** Streamlit Cloud installs dependencies from `requirements.txt` automatically. Ensure `requirements.txt` is present at the repository root (included in this repo).
- **Create the app:** Go to https://share.streamlit.io, connect your GitHub account, select the `kiran86/realestate-advisor` repo and the `main` branch, then set the app path to `streamlit_app/app.py`.
- **Advanced settings:** Add any environment secrets via the Streamlit Cloud dashboard (for API keys or private credentials).

Quick checklist:

- **Push changes:** `git add . && git commit -m "Add Streamlit Cloud support" && git push`
- **Select repo & branch:** use the Streamlit Cloud UI to point to `streamlit_app/app.py`.
- **Wait for build:** Streamlit Cloud will install packages and launch the app; monitor logs for dependency issues.

If `geopandas` or other geo packages cause installation issues on Streamlit Cloud, consider pre-processing geo files into plain GeoJSON and using `folium` or `pydeck` to visualize them.

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to open a pull request or contact the project owner.

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.


---

&copy; 2026 RealEstate Advisor Project by kiran86. All rights reserved.