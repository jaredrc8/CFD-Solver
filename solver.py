# 2D CFD Solver for Euler Equations of Airflow over Inclined Flat Plate
# Incompressible, Laminar, Steady-State, Finite Volume Method
# Time Schemes: Forward Euler
# Solution Schemes: Central Difference, First-Order Upwind

# Author: Jared Crebo, 2025

# Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
from cell import cell

# Parameters of Inclined Flat Plate
plate_incline = 1 # degrees
plate_length = 1 # meters

# Parameters of Flow
airspeed = 5 # m/s

# Parameters of Domain
domain_x = plate_length * 5 # meters
domain_y = plate_length * 2 # meters
nx = 50 # Number of Nodes in x-direction
ny = 20 # Number of Nodes in y-direction
cx = nx - 1 # Number of Cells in x-direction
cy = ny - 1 # Number of Cells in y-direction

# Generate Structured Mesh with nx x ny Cells
def mesh_generation():
    # Generate Mesh Nodes
    x = np.linspace(-domain_x / 2, domain_x / 2, nx)
    y = np.linspace(-domain_y / 2, domain_y / 2, ny)
    nodes_x, nodes_y = np.meshgrid(x, y)

    # Calculate Cell Centers
    xc = 0.5 * (nodes_x[1:, 1:] + nodes_x[:-1, :-1])
    yc = 0.5 * (nodes_y[1:, 1:] + nodes_y[:-1, :-1])
    centers_x, centers_y = np.meshgrid(xc, yc)

    # Generate Mesh Cell Objects
    mesh = np.empty((cx, cy), dtype=object)
    for i in range(cx):
        for j in range(cy):
            mesh[i,j] = cell(centers_x[i,j], centers_y[i,j], domain_x, domain_y, nx, ny)
    
    # Assign Cell Neighbours for Each Cell Object
    for i in range(cx):
        for j in range(cy):
            Cell = mesh[i,j]
            try:
                if j < cy-1:
                    north = [i,j+1]
                else:
                    north = None
                if j > 0:
                    south = [i,j-1]
                else:
                    south = None
                if i < cx-1:
                    east = [i+1,j]
                else:
                    east = None
                if i > 0:
                    west = [i-1,j]
                else:
                    west = None
            except IndexError:
                print("IndexError at [" + str(i) + "," + str(j) + "]")
            cell.set_neighbours(Cell, north, south, east, west)

    return nodes_x, nodes_y, centers_x, centers_y, mesh

def set_boundary_conditions():
    # Vertical Boundaries
    for j in range(cy):
        # Inlet Boundary Condition (Left Wall)
        mesh[0,j].is_boundary = True
        mesh[0,j].boundary_type = "inlet"
        
        # Outlet Boundary Condition (Right Wall)
        mesh[cx-1,j].is_boundary = True
        mesh[cx-1,j].boundary_type = "outlet"
    
    # Horizontal Boundaries
    for i in range(cx):
        # Lower Freestream Boundary Condition (Bottom Wall)
        mesh[i,0].is_boundary = True
        mesh[i,0].boundary_type = "freestream"

        # Upper Freestream Boundary Condition (Top Wall)
        mesh[i,cy-1].is_boundary = True
        mesh[i,cy-1].boundary_type = "freestream"


def set_plate():

    return None


nodes_x, nodes_y, centers_x, centers_y, mesh = mesh_generation()
set_boundary_conditions()

# Plot Mesh
fig, ax = plt.subplots(figsize=(10, 4))
plt.scatter(nodes_x, nodes_y, color='red', s=10, zorder=5, label="Grid points")
plt.scatter(centers_x, centers_y, color='blue', s=10, zorder=5, label="Cell centers")
for i in range(cx):
    for j in range(cy):
        if mesh
        rect = patches.Rectangle(xy=(nodes_x[j,i], nodes_y[j,i]), width=mesh[i,j].length_x, height=mesh[i,j].length_y, 
                                linewidth=1, edgecolor='black', facecolor='lightgrey')
        ax.add_patch(rect)
ax.set_xlim(-domain_x / 2, domain_x / 2)
ax.set_ylim(-domain_y / 2, domain_y / 2)
# plt.show()



