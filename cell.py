# Object containing the parameters of a cell in the grid
class cell:
    def __init__(self, Xc, Yc, domain_x, domain_y, nx, ny):
        self.Xc = Xc # X-coordinate of cell center
        self.Yc = Yc # Y-coordinate of cell center
        self.length_x = domain_x / nx # Length of cell in x-direction
        self.length_y = domain_y / ny # Length of cell in y-direction
        self.volume = self.length_x * self.length_y # Volume of cell

        # Flow variables
        self.rho = 0.0 # Density
        self.u = 0.0  # Velocity in x-direction
        self.v = 0.0  # Velocity in y-direction
        self.p = 0.0  # Pressure

        # Neighbor indices (or references)
        self.neighbours = {"N": None, "S": None, "E": None, "W": None}

        # Fluxes (per face)
        self.flux_u = {"N": 0.0, "S": 0.0, "E": 0.0, "W": 0.0}
        self.flux_v = {"N": 0.0, "S": 0.0, "E": 0.0, "W": 0.0}
        self.flux_p = {"N": 0.0, "S": 0.0, "E": 0.0, "W": 0.0}

        # Boundary flag
        self.is_boundary = False
        self.boundary_type = None  # e.g., "inlet", "outlet", "wall"
    
    def set_neighbours(self, north, south, east, west):
        self.neighbours["N"] = north
        self.neighbours["S"] = south
        self.neighbours["E"] = east
        self.neighbours["W"] = west