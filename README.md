# VRP Simulate 🚚📦

A compact complete VRP Simulate simulator was create to optimise **Vehicle Routing Problem (VRP)**.

---

## 📑 Key Features

| Module               | Description                                                                                   |
| -------------------- | --------------------------------------------------------------------------------------------- |
| `DESGreedySimulator` | Event-driven, capacity-aware greedy dispatcher that creates an initial set of routes          |
| `RouteImprover`      | K-Means clustering ➜ Clarke-Wright Savings ➜ 2-opt local search to reduce distance & vehicles |
| Visualiser           | Simple Matplotlib plots (`routes_initial.png` / `routes_improved.png`)                        |
| Sample data          | `orders.csv` (20-row sample) + script/snippet to generate any number of random orders         |

---

## 🏗️ Project Structure

```
vrp_simulate/
├── vrp_simulation.py        # main script (simulator + optimiser + plots)
├── orders.csv               # sample data (20 rows) – replace with your own
├── requirements.txt         # lightweight dependency list
├── README.md                # this file
└── .gitignore               # venv, cache, output artefacts, etc.
```

---

## ⚙️ Quick Start

```bash
# 1) Clone the repo
 git clone https://github.com/marcelonbarreto/vrp_simulate.git
 cd vrp_simulate

# 2) Create and activate a virtual env (recommended)
 python3 -m venv venv
 source venv/bin/activate  # Windows: venv\Scripts\activate

# 3) Install dependencies
 pip install -r requirements.txt

# 4) Run the simulator (PNG plots will be generated)
 python vrp_simulation.py --plot
```

### CLI options

| Argument   | Default      | Description                                       |
| ---------- | ------------ | ------------------------------------------------- |
| `--orders` | `orders.csv` | Path to CSV *or* Excel file with orders           |
| `--plot`   | off          | Save `routes_initial.png` & `routes_improved.png` |

---

## 📥 Order File Format

A plain UTF-8 CSV (or `.xlsx`) with **exact column headers**:

```text
order_id,x,y,qty_A,qty_B
1,  -28.58,-24.21,3,12
2,  60.83,73.06,4,1
…
```

* `(x, y)` are kilometres from the warehouse `(0, 0)`.
* **Volume formula**: `vol = 3 × qty_A + 1 × qty_B` → compared to truck capacity.

Need dummy data?  Generate N orders on the fly:

```bash
python - <<'PY'
import pandas as pd, numpy as np, sys, math
N = int(sys.argv[1]) if len(sys.argv)>1 else 1000
r = np.random.uniform(0, 100, N)
a = np.random.uniform(0, 2*math.pi, N)
orders = pd.DataFrame({
    'order_id': range(1, N+1),
    'x': np.round(r*np.cos(a), 2),
    'y': np.round(r*np.sin(a), 2),
    'qty_A': np.random.randint(0,5,N),
    'qty_B': np.random.randint(0,15,N)
})
orders.to_csv('orders.csv', index=False)
print('Generated', N, 'orders → orders.csv')
PY 500
```

---

## 📊 Output Files

| File                    | Purpose                                                                                            |
| ----------------------- | -------------------------------------------------------------------------------------------------- |
| `initial_solution.csv`  | Each row = one stop. Columns: route ID, vehicle type, stop sequence, order ID, coords & quantities |
| `improved_solution.csv` | Same as above but after optimisation                                                               |
| `routes_initial.png`    | Visual diagram of greedy routes                                                                    |
| `routes_improved.png`   | Visual diagram of                                                                                  |

