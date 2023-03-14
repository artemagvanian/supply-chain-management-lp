from __future__ import annotations

import dataclasses
import json
import math
from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt
import networkx as nx
from docplex.mp.model import *

VISUALIZE = False


@dataclass()
class LPInstance:
    num_customers: int  # the number of customers
    num_facilities: int  # the number of facilities
    alloc_cost_cf: list[list[float]]  # allocCostCF[c][f] is the service cost paid each time c is served by f
    demand_c: list[float]  # demandC[c] is the demand of customer c
    opening_cost_f: list[float]  # openingCostF[f] is the opening cost of facility f
    capacity_f: list[float]  # capacityF[f] is the capacity of facility f
    num_max_vehicle_per_facility: int  # maximum number of vehicles to use at an open facility
    truck_dist_limit: float  # total driving distance limit for trucks
    truck_usage_cost: float  # fixed usage cost paid if a truck is used
    distance_cf: list[list[float]]  # distanceCF[c][f] is the roundtrip distance between customer c and facility f


def get_lp_instance(file_name: str) -> Optional[LPInstance]:
    try:
        with open(file_name, "r") as fl:
            num_customers, num_facilities = [int(i) for i in fl.readline().split()]
            num_max_vehicle_per_facility = num_customers
            print(f"num_customers: {num_customers} " +
                  f"numFacilities: {num_facilities} " +
                  f"numVehicle: {num_max_vehicle_per_facility}")

            # noinspection DuplicatedCode
            alloc_cost_cf = [[0.0 for _ in range(num_facilities)] for _ in range(num_customers)]
            alloc_cost_raw = [float(i) for i in fl.readline().split()]
            index = 0
            for i in range(num_customers):
                for j in range(num_facilities):
                    alloc_cost_cf[i][j] = alloc_cost_raw[index]
                    index += 1

            demandC = [float(i) for i in fl.readline().split()]
            opening_cost_f = [float(i) for i in fl.readline().split()]
            capacity_f = [float(i) for i in fl.readline().split()]
            truck_dist_limit, truck_usage_cost = [float(i) for i in fl.readline().split()]

            # noinspection DuplicatedCode
            distance_cf = [[0.0 for _ in range(num_facilities)] for _ in range(num_customers)]
            distance_cf_raw = [float(i) for i in fl.readline().split()]
            index = 0
            for i in range(num_customers):
                for j in range(num_facilities):
                    distance_cf[i][j] = distance_cf_raw[index]
                    index += 1

            return LPInstance(
                num_customers=num_customers,
                num_facilities=num_facilities,
                alloc_cost_cf=alloc_cost_cf,
                demand_c=demandC,
                opening_cost_f=opening_cost_f,
                capacity_f=capacity_f,
                num_max_vehicle_per_facility=num_max_vehicle_per_facility,
                truck_dist_limit=truck_dist_limit,
                truck_usage_cost=truck_usage_cost,
                distance_cf=distance_cf
            )

    except Exception as e:
        print(f"Could not read problem instance file due to error: {e}")
        return None


class LPSolver:

    def __init__(self, filename: str):
        self.filename = filename
        self.lp_instance = get_lp_instance(filename)
        self.model = Model()  # CPLEX solver
        self.fc_matrix = []
        self.build_constraints()

    def build_constraints(self):
        # Create an FC-matrix -- fc_matrix[facility][customer]
        for i in range(self.lp_instance.num_facilities):
            f_row = []
            for j in range(self.lp_instance.num_customers):
                f_row.append(self.model.continuous_var(0, name=f"F{i}C{j}"))
            self.fc_matrix.append(f_row)

        # Enforce capacity constraints
        for i in range(self.lp_instance.num_facilities):
            self.model.add_constraint(self.model.sum(self.fc_matrix[i]) <= self.lp_instance.capacity_f[i])

        # Enforce supply == demand constraints
        for i in range(self.lp_instance.num_customers):
            c_col = [f[i] for f in self.fc_matrix]
            self.model.add_constraint(self.model.sum(c_col) == self.lp_instance.demand_c[i])

        # Enforce total distance constraints (linear relaxation of the truck constraint)
        for i in range(self.lp_instance.num_facilities):
            # Multiply distance by the proportion of demand served
            distances = [self.fc_matrix[i][j] * self.lp_instance.distance_cf[j][i] / self.lp_instance.demand_c[j]
                         for j in range(self.lp_instance.num_customers)]
            # Total distance is <= than the maximum number of trucks * truck distance limit
            self.model.add_constraint(self.model.sum(distances) <=
                                      self.lp_instance.num_max_vehicle_per_facility * self.lp_instance.truck_dist_limit)

        # Calculate opening cost as a proportion of capacity used
        opening_cost = self.model.sum([
            self.model.sum(self.fc_matrix[i]) / self.lp_instance.capacity_f[i] * self.lp_instance.opening_cost_f[i]
            for i in range(self.lp_instance.num_facilities)])

        # Calculate service cost as a proportion of customer demand
        service_cost = 0
        for i in range(self.lp_instance.num_facilities):
            for j in range(self.lp_instance.num_customers):
                service_cost += \
                    self.lp_instance.alloc_cost_cf[j][i] * self.fc_matrix[i][j] / self.lp_instance.demand_c[j]

        # Calculate truck usage cost as a (not necessarily integer) number of trucks used
        truck_usage_cost = self.model.sum([
            self.model.sum(self.fc_matrix[i]) / self.lp_instance.truck_dist_limit * self.lp_instance.truck_usage_cost
            for i in range(self.lp_instance.num_facilities)])

        total_cost = opening_cost + service_cost + truck_usage_cost

        self.model.minimize(total_cost)

    def solve(self):
        solution = self.model.solve()
        if VISUALIZE:
            self.model.print_information()
            print(self.model.pprint_as_string())
            self.visualize_lp(solution)
        return self.model.objective_value

    def visualize_lp(self, solution):
        G = nx.Graph()
        for i in range(self.lp_instance.num_facilities):
            for j in range(self.lp_instance.num_customers):
                if solution[self.fc_matrix[i][j]] != 0:
                    G.add_node(f"F{i}", color="red")
                    G.add_node(f"C{j}", color="blue")
                    G.add_edge(f"F{i}", f"C{j}", label=f"{solution[self.fc_matrix[i][j]]}")

        plt.figure(figsize=(10, 10), dpi=300)

        pos = nx.nx_pydot.graphviz_layout(G)

        nx.draw(G, pos, with_labels=True, font_size=8, font_color="white",
                node_color=[d['color'] for n, d in G.nodes(data=True)])

        edge_labels = dict([((n1, n2), f'{d["label"]}')
                            for n1, n2, d in G.edges(data=True)])
        nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=6)

        plt.savefig(f"visualizations/{os.path.basename(self.filename)}.png", dpi="figure")
        with open(f"visualizations/{os.path.basename(self.filename)}.json", "w") as f:
            f.write(json.dumps(dataclasses.asdict(self.lp_instance)))


def diet_problem():
    # Diet Problem from Lecture Notes
    m = Model()
    # Note that these are continuous variables and not integers
    model_vars = m.continuous_var_list(2, 0, 1000)
    carbs = m.scal_prod(terms=model_vars, coefs=[100, 250])
    m.add_constraint(carbs >= 500)

    m.add_constraint(m.scal_prod(terms=model_vars, coefs=[100, 50]) >= 250)  # Fat
    m.add_constraint(m.scal_prod(terms=model_vars, coefs=[150, 200]) >= 600)  # Protein

    m.minimize(m.scal_prod(terms=model_vars, coefs=[25, 15]))

    sol = m.solve()
    obj_value = math.ceil(m.objective_value)
    if sol:
        m.print_information()
        print(f"Meat: {model_vars[0].solution_value}")
        print(f"Bread: {model_vars[1].solution_value}")
        print(f"Objective Value: {obj_value}")
    else:
        print("No solution found!")
