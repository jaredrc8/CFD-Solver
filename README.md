Personal project by Jared Crebo 2025 to implement a solver of the 2D Euler equations for airflow over an inclined flat plate. 
The mesh is created as a structured grid of cell objects (defined by cell.py) that contain all the data for each cell. 
The boundary conditions and initial conditions are applied to the flat plate by identifying which indices of cells are at the locations that make a straight line at a specified inclination. 

Assumptions: incompressible, laminar, steady-state flow

Method: Finite Volume Method

Time Schemes: Forward Euler

Solution Schemes: Central Difference (convection)
