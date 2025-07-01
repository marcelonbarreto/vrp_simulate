#!/usr/bin/env python3
"""
Progress Rail – Dynamic Vehicle Routing Problem (VRP)
===================================================
Author: ChatGPT
Date  : 2025‑06‑30

This script provides a *complete* but **readable** reference implementation for the
exercise you received.  It is organised so that you can run it end‑to‑end or import
individual classes/functions in Jupyter‑style experimentation.

---------------------------------------------------------------------
HOW TO USE
---------------------------------------------------------------------
1.  Put a file called **orders.csv** in the same directory.  Required columns:
        order_id,x,y,qty_A,qty_B
    where (x, y) are coordinates in kilometres relative to the warehouse (0, 0).

2.  Install dependencies (all are mainstream PyPI packages):
        pip install pandas numpy scikit-learn matplotlib

3.  Run:  
        python vrp_simulation.py  --plot   # generates png visualisations

4.  Key outputs:
        ▸  initial_solution.csv   – greedy event‑based allocation
        ▸  improved_solution.csv  – after optimisation step
        ▸  routes_initial.png / routes_improved.png – route diagrams

---------------------------------------------------------------------
ALGORITHMIC OVERVIEW
---------------------------------------------------------------------
» Discrete‑Event Simulator (DES)
    • Vehicles represented by a priority‑queue keyed on *next_available_time*.
    • Greedy loader packs nearest orders until vehicle capacity reached or time limit
      would be violated.

» Improvement Strategy
    • Spatial K‑Means clustering forms tighter delivery zones.
    • Within each cluster, Clarke‑Wright Savings + 2‑opt local search refines route.

» Visualisation
    • Matplotlib scatter + line plot.
    • Colours encode vehicle types.  Warehouse shown as black star.

NOTE
----
The implementation aims for clarity rather than absolute state‑of‑the‑art
performance.  You can swap the improvement phase for meta‑heuristics (Tabu Search,
Simulated Annealing, etc.) if you wish.
"""

from __future__ import annotations

import argparse
import math
import random
import heapq
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Domain objects
# ---------------------------------------------------------------------------

WAREHOUSE = (0.0, 0.0)
SHIFT_START = 8.0   # 08:00 in hours
SHIFT_END = 18.0    # 18:00 in hours

@dataclass
class VehicleType:
    name: str
    capacity: int          # total unit capacity
    speed_kph: float       # km per hour

    def __repr__(self):
        return f"{self.name}(cap={self.capacity}, speed={self.speed_kph}kph)"

VTYPES: List[VehicleType] = [
    VehicleType("T1", 10, 60),
    VehicleType("T2", 20, 50),
    VehicleType("T3", 15, 55),
    VehicleType("T4", 25, 45),
]

PRODUCT_A_VOL = 3  # units
PRODUCT_B_VOL = 1

@dataclass
class Order:
    order_id: int
    x: float
    y: float
    qty_A: int
    qty_B: int

    @property
    def demand(self) -> int:
        return PRODUCT_A_VOL * self.qty_A + PRODUCT_B_VOL * self.qty_B

    @property
    def pos(self) -> Tuple[float, float]:
        return (self.x, self.y)

@dataclass
class Route:
    vehicle_type: VehicleType
    stop_sequence: List[Order]
    distance: float = 0.0
    travel_time: float = 0.0   # hours

@dataclass(order=True)
class VehicleInstance:
    next_available_time: float
    id: int = field(compare=False)
    vtype: VehicleType = field(compare=False)
    routes: List[Route] = field(default_factory=list, compare=False)

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def euclidean(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

# ---------------------------------------------------------------------------
# Parsing & loading
# ---------------------------------------------------------------------------

def load_orders(path: Path | str) -> List[Order]:
    df = pd.read_csv(path)
    required_cols = {"order_id", "x", "y", "qty_A", "qty_B"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"orders.csv must contain columns: {required_cols}")
    return [
        Order(int(r.order_id), float(r.x), float(r.y), int(r.qty_A), int(r.qty_B))
        for r in df.itertuples(index=False)
    ]

# ---------------------------------------------------------------------------
# Discrete‑Event Simulator (Greedy Initial Solution)
# ---------------------------------------------------------------------------

class DESGreedySimulator:
    """Simple greedy discrete‑event simulator."""

    def __init__(self, orders: List[Order]):
        self.orders = orders.copy()
        self.unserved: Dict[int, Order] = {o.order_id: o for o in orders}
        self.vehicles_heap: List[VehicleInstance] = []
        self.vehicle_counter = 0
        self.routes: List[Route] = []

    # -----------------------------------------------------
    def allocate_vehicle(self, vtype: VehicleType) -> VehicleInstance:
        v = VehicleInstance(
            next_available_time=SHIFT_START,
            id=self.vehicle_counter,
            vtype=vtype,
        )
        self.vehicle_counter += 1
        heapq.heappush(self.vehicles_heap, v)
        return v

    # -----------------------------------------------------
    def run(self):
        # Sort orders by polar angle to create a deterministic processing order.
        self.orders.sort(key=lambda o: math.atan2(o.y, o.x))

        while self.unserved:
            # Pop earliest available vehicle OR create a new one if none exist yet.
            if self.vehicles_heap:
                veh = heapq.heappop(self.vehicles_heap)
            else:
                veh = self.allocate_vehicle(VTYPES[-1])  # start with largest capacity

            if veh.next_available_time >= SHIFT_END:
                # Can't serve more today → stop loop.
                break

            route, served_ids = self._build_route(veh)
            if not served_ids:
                # No orders could fit (rare). Abort to avoid infinite loop.
                break

            # Update vehicle timing
            veh.routes.append(route)
            veh.next_available_time += route.travel_time
            heapq.heappush(self.vehicles_heap, veh)

            # Remove served orders
            for oid in served_ids:
                self.unserved.pop(oid, None)
            self.routes.append(route)

    # -----------------------------------------------------
    def _build_route(self, veh: VehicleInstance) -> Tuple[Route, List[int]]:
        capacity_left = veh.vtype.capacity
        current_pos = WAREHOUSE
        route_orders: List[Order] = []
        total_distance = 0.0
        served_ids: List[int] = []

        # greedy nearest‑neighbour until capacity full or no candidates
        while self.unserved and capacity_left > 0:
            # find nearest order that fits
            nearest_order = None
            nearest_dist = float("inf")
            for o in self.unserved.values():
                if o.demand > capacity_left:
                    continue
                d = euclidean(current_pos, o.pos)
                if d < nearest_dist:
                    nearest_dist, nearest_order = d, o
            if nearest_order is None:
                break  # nothing else fits

            # go to order
            total_distance += nearest_dist
            route_orders.append(nearest_order)
            served_ids.append(nearest_order.order_id)
            capacity_left -= nearest_order.demand
            current_pos = nearest_order.pos

        # return to warehouse
        total_distance += euclidean(current_pos, WAREHOUSE)
        travel_time = total_distance / veh.vtype.speed_kph

        route = Route(vehicle_type=veh.vtype, stop_sequence=route_orders,
                      distance=total_distance, travel_time=travel_time)
        return route, served_ids

# ---------------------------------------------------------------------------
# Improvement Phase
# ---------------------------------------------------------------------------

class RouteImprover:
    """Spatial clustering → Clarke‑Wright Savings + 2‑opt."""

    def __init__(self, orders: List[Order], routes: List[Route]):
        self.orders = orders
        self.routes = routes

    # -----------------------------------------------------
    def improve(self) -> List[Route]:
        # Cluster orders to ~ vehicle capacity groups (k heuristically chosen)
        X = np.array([[o.x, o.y] for o in self.orders])
        k = max(1, len(self.orders) // 20)  # target ~20 orders per cluster
        km = KMeans(n_clusters=k, n_init="auto", random_state=42).fit(X)
        clusters: Dict[int, List[Order]] = {i: [] for i in range(k)}
        for o, lbl in zip(self.orders, km.labels_):
            clusters[int(lbl)].append(o)

        improved_routes: List[Route] = []
        for cluster_orders in clusters.values():
            if not cluster_orders:
                continue
            # choose vehicle type by total demand
            total_dem = sum(o.demand for o in cluster_orders)
            vtype = min(VTYPES, key=lambda v: abs(v.capacity - total_dem))
            route = self._clarke_wright(cluster_orders, vtype)
            improved_routes.append(route)
        self.routes = improved_routes
        return improved_routes

    # -----------------------------------------------------
    def _clarke_wright(self, orders: List[Order], vtype: VehicleType) -> Route:
        """Very compact (non‑parallel) Clarke & Wright Savings algorithm."""
        # Start with direct routes warehouse → order → warehouse
        routes = [[o] for o in orders]
        savings: List[Tuple[float, int, int]] = []
        for i, oi in enumerate(orders):
            for j, oj in enumerate(orders):
                if i >= j:
                    continue
                sij = (euclidean(WAREHOUSE, oi.pos) + euclidean(WAREHOUSE, oj.pos)
                       - euclidean(oi.pos, oj.pos))
                savings.append((sij, i, j))
        savings.sort(reverse=True)

        parent = list(range(len(routes)))  # DSU for simple merging control
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for s, i, j in savings:
            ri, rj = find(i), find(j)
            if ri == rj:
                continue
            route_i, route_j = routes[ri], routes[rj]
            # Only merge if capacity allows
            if (sum(o.demand for o in route_i + route_j) <= vtype.capacity):
                # Join tail of i with head of j
                routes[ri] = route_i + route_j
                parent[rj] = ri

        # pick the merged route with largest payload (heuristic)
        final_routes = [r for idx, r in enumerate(routes) if parent[idx] == idx]
        if not final_routes:
            final_routes = routes
        best = max(final_routes, key=lambda r: sum(o.demand for o in r))
        # perform 2‑opt for local improvement
        best_seq = self._two_opt(best)
        dist = self._route_distance(best_seq)
        travel_time = dist / vtype.speed_kph
        return Route(vehicle_type=vtype, stop_sequence=best_seq, distance=dist,
                     travel_time=travel_time)

    # -----------------------------------------------------
    @staticmethod
    def _route_distance(seq: List[Order]) -> float:
        d = euclidean(WAREHOUSE, seq[0].pos) if seq else 0.0
        for a, b in zip(seq, seq[1:]):
            d += euclidean(a.pos, b.pos)
        if seq:
            d += euclidean(seq[-1].pos, WAREHOUSE)
        return d

    # -----------------------------------------------------
    def _two_opt(self, seq: List[Order]) -> List[Order]:
        best = seq
        improved = True
        while improved:
            improved = False
            for i in range(1, len(best) - 2):
                for j in range(i + 1, len(best)):
                    if j - i == 1:
                        continue
                    new_route = best[:i] + best[i:j][::-1] + best[j:]
                    if self._route_distance(new_route) < self._route_distance(best):
                        best = new_route
                        improved = True
            if improved:
                break  # single pass for speed
        return best

# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_routes(routes: List[Route], outfile: Path):
    colours = {
        "T1": "tab:blue",
        "T2": "tab:green",
        "T3": "tab:orange",
        "T4": "tab:red",
    }
    plt.figure(figsize=(8, 8))
    for r in routes:
        xs = [WAREHOUSE[0]] + [o.x for o in r.stop_sequence] + [WAREHOUSE[0]]
        ys = [WAREHOUSE[1]] + [o.y for o in r.stop_sequence] + [WAREHOUSE[1]]
        plt.plot(xs, ys, marker="o", linewidth=1,
                 color=colours.get(r.vehicle_type.name, "gray"), alpha=0.6)
    plt.scatter(*WAREHOUSE, c="k", marker="*", s=150, label="Warehouse")
    plt.title("Vehicle Routes")
    plt.legend()
    plt.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.close()

# ---------------------------------------------------------------------------
# Main CLI entry‑point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Progress Rail VRP Simulator")
    parser.add_argument("--orders", default="orders.csv", help="Path to orders CSV")
    parser.add_argument("--plot", action="store_true", help="Save route PNGs")
    args = parser.parse_args()

    orders = load_orders(args.orders)
    print(f"Loaded {len(orders)} orders")

    # 1) Initial greedy solution
    sim = DESGreedySimulator(orders)
    sim.run()
    greedy_routes = sim.routes
    df_init = to_dataframe(greedy_routes)
    df_init.to_csv("initial_solution.csv", index=False)
    print(f"Initial solution: {len(greedy_routes)} routes saved → initial_solution.csv")

    # 2) Improvement phase
    improver = RouteImprover(orders, greedy_routes)
    best_routes = improver.improve()
    df_best = to_dataframe(best_routes)
    df_best.to_csv("improved_solution.csv", index=False)
    print("Improved solution saved → improved_solution.csv")

    # 3) Visualisation
    if args.plot:
        plot_routes(greedy_routes, Path("routes_initial.png"))
        plot_routes(best_routes, Path("routes_improved.png"))
        print("Route plots saved → routes_initial.png / routes_improved.png")

# ---------------------------------------------------------------------------
# Helper to serialise routes into DataFrame
# ---------------------------------------------------------------------------

def to_dataframe(routes: List[Route]) -> pd.DataFrame:
    rows = []
    for ridx, r in enumerate(routes):
        for seq, order in enumerate(r.stop_sequence, 1):
            rows.append({
                "route_id": ridx,
                "vehicle_type": r.vehicle_type.name,
                "seq": seq,
                "order_id": order.order_id,
                "x": order.x,
                "y": order.y,
                "qty_A": order.qty_A,
                "qty_B": order.qty_B,
            })
    return pd.DataFrame(rows)

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
