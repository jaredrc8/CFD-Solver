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
plate_incline = 30 # degrees
plate_length = 0.5 # meters

# Parameters of Flow
airspeed = 5 # m/s
density = 1.225 # kg/m^3

# Parameters of Domain
domain_x = plate_length * 10 # meters
domain_y = plate_length * 4 # meters
nx = 150 # Number of Nodes in x-direction
ny = 60 # Number of Nodes in y-direction
cx = nx - 1 # Number of Cells in x-direction
cy = ny - 1 # Number of Cells in y-direction
cell_length_x = domain_x / nx # Length of Cell in x-direction
cell_length_y = domain_y / ny # Length of Cell in y-direction

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

# Set Boundary Conditions on Outer Bounds of Domain
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

# Set Boundary Conditions on Object in Flow
def set_object():
    # Starting Point (0,0)
    mesh[locate_cell_index(0,0)].is_boundary = True
    mesh[locate_cell_index(0,0)].boundary_type = "noSlip"

    # Plate Boundary Conditions
    x = 0
    y = 0
    for i in range(0,int(plate_length/2 * 100),int(round(cell_length_x/2 * 100))):
        # Identify (x,y) Location of Plate
        x += cell_length_x * np.cos(np.deg2rad(plate_incline))
        y += cell_length_y * np.sin(np.deg2rad(plate_incline))

        # Identify Which Cell is at that Location and Set Boundary Condition
        a, b = locate_cell_index(x, y)
        mesh[a,b].is_boundary = True
        mesh[a,b].boundary_type = "noSlip"
        d, e = locate_cell_index(-x, -y)
        mesh[d,e].is_boundary = True
        mesh[d,e].boundary_type = "noSlip"

# Return Index of Cell in (x,y) Location
def locate_cell_index(x, y):
    index_x = 0
    index_y = 0
    for i in range(cx-1):
        if x <= nodes_x[0,i+1] and x >= nodes_x[0,i]:
            index_x = i
    for j in range(cy-1):
        if y <= nodes_y[j+1,0] and y >= nodes_y[j,0]:
            index_y = j
    return index_x, index_y

def locate_cell_location(i, j):
    return mesh[i,j].Xc, mesh[i,j].Yc

def set_initial_conditions():
    for i in range(cx):
        for j in range(cy):
            if mesh[i,j].is_boundary == False:
                mesh[i,j].rho = density
                mesh[i,j].u = airspeed
                mesh[i,j].v = 0.0
                mesh[i,j].p = 0.0
            else:
                if mesh[i,j].boundary_type == "inlet" or mesh[i,j].boundary_type == "outlet" or mesh[i,j].boundary_type == "freestream":
                    mesh[i,j].rho = density
                    mesh[i,j].u = airspeed
                    mesh[i,j].v = 0.0
                    mesh[i,j].p = 0.0
                elif mesh[i,j].boundary_type == "noSlip":
                    mesh[i,j].rho = density
                    mesh[i,j].u = 0.0
                    mesh[i,j].v = 0.0
                    mesh[i,j].p = 0.0
                else:
                    print("Boundary Condition Error at [" + str(mesh[i,j].Xc) + "," + str(mesh[i,j].Yc) + "]")

def compute_fluxes(cell):
    # Compute Convective and Pressure Fluxes Across a Face
    # First-Order Upwind Scheme

    # Flow Variables of Current Cell
    u = cell.u
    v = cell.v
    p = cell.p

    directions = ["N", "S", "E", "W"]
    for direction in directions:
        # Cell Neighbour in Direction
        neighbour = mesh[cell.neighbours[direction]]
        if neighbour is not None:
            if direction == "N":
                if v > 0: # Flow from current cell to neighbour
                    phi_upwind = v
                else: # Flow from neighbour to current cell
                    phi_upwind = neighbour.v
                cell.flux_v["N"] = v * phi_upwind * cell.volume


def assemble_matrix():
    A = np.zeros(cx*cy, cx*cy)
    b = np.zeros(cx*cy)
    for cell in mesh:
        i = mesh.index(cell)
        diagonal_coeffs = 0.0
        for neighbours in cell.neighbours:
            if neighbours is not None:
                face_flux = 0

    return A, b

nodes_x, nodes_y, centers_x, centers_y, mesh = mesh_generation()
set_boundary_conditions()
set_object()
set_initial_conditions()

# Plot Mesh
fig, ax = plt.subplots(figsize=(10, 4))
#plt.scatter(nodes_x, nodes_y, color='red', s=10, zorder=5, label="Grid points")
#plt.scatter(centers_x, centers_y, color='blue', s=10, zorder=5, label="Cell centers")
for i in range(cx):
    for j in range(cy):
        if mesh[i,j].is_boundary == True:
            rect = patches.Rectangle(xy=(nodes_x[j,i], nodes_y[j,i]), width=mesh[i,j].length_x, height=mesh[i,j].length_y, 
                                    linewidth=1, edgecolor='black', facecolor='black')
        else:
            rect = patches.Rectangle(xy=(nodes_x[j,i], nodes_y[j,i]), width=mesh[i,j].length_x, height=mesh[i,j].length_y, 
                                    linewidth=1, edgecolor='darkgrey', facecolor='white')
        ax.add_patch(rect)
ax.set_xlim(-domain_x / 2, domain_x / 2)
ax.set_ylim(-domain_y / 2, domain_y / 2)
plt.show()

