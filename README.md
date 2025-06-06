# UFC match prediction ML model

## Problem statement

UFC (Ultimate Fighting Championship) este o organizație globală de arte marțiale mixte (MMA), care găzduiește evenimente competitive săptămânale ce prezintă luptători din diverse categorii de greutate și origini.
Acest model are ca scop prezicerea meciurilor de MMA din cadrul UFC. Acest proiect include:
- model cached pentru a evita timpul lung de așteptare
- decision analysis: ne arată criteriile după care s-a luat modelul pentru a lua decizia

## Video Demo

[link youtube](https://www.youtube.com/watch?v=BmU4HpJyMO4)

## Data set

- /data/complete_ufc_data.csv  date despre meciurile UFC din ultimii 31 de ani (din 1994)

## Setup

### 1. **Prima oară** (Train & Cache Model)
```bash
# Setup the model (takes 2-3 minutes first time)
python instant_predict.py --setup
```

### 2. **Predictii instante (CLI)**
```bash
# Quick prediction
python instant_predict.py "Jon Jones" "Stipe Miocic"

# With detailed explanation
python instant_predict.py "Jon Jones" "Stipe Miocic" --explain

# Interactive mode
python instant_predict.py --interactive
```

### 3. **Analiza detaliata**
```bash
# Full interactive system with decision metrics
python cached_prediction_tool.py
```

## 📦 Installation

### **Prerequisites**
```bash
pip install pandas numpy scikit-learn xgboost requests beautifulsoup4
```

### **Project Setup**
```bash
# Clone or download the project
git clone <your-repo-url>
cd ufc-prediction

# Ensure your data file is in place
# data/complete_ufc_data.csv (31 years of UFC data)

# First time setup
python instant_predict.py --setup
```

## made by mihai * (bujor + buga)
