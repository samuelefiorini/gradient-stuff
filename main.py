import sys
import os

from gradient_stuff.viz.projection_viz import visualize_projection
from gradient_stuff.viz.boosting_viz import generate_boosting_visualizations

def main():
    print("Generating Gradient Boosting Visualizations...")
    os.makedirs("assets", exist_ok=True)
    generate_boosting_visualizations()
    
    print("Generating Simplex Projection Visualization...")
    visualize_projection(filename="assets/simplex_projection.gif")

if __name__ == "__main__":
    main()
