import numpy as np
import matplotlib.pyplot as plt
import alphashape


def getboundary(points, alpha=0.0):
    hull = alphashape.alphashape(points, alpha)
    boundary_points = np.asarray(hull.exterior.coords)
    return boundary_points
