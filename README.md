# ML4MDE project

This repository contains the code and the data for the ML4MDE project.

## Project structure

The project is structured as follows:

- `archive/`: contains the data used for the project
- `PaperClassification.ipynb`: contains the code for the paper classification task
- `dashboard.py`: contains the code for the dashboard
- `vectorization.pkl`: contains the vectorization model
- `paper_classification.keras`: contains the trained model for the paper classification task
- `requirements.txt`: contains the required packages for the project
- `README.md`: contains the description of the project

## How to run the dashboard

In order to run the dashboard you need to install the required packages. You can do this by running the following command:

```bash
pip install -r requirements.txt
```

After that, you can run the dashboard by running the following command:

```bash
python dashboard.py
```

The dashboard will be available at the following address: [http://localhost:8050](http://localhost:8050)