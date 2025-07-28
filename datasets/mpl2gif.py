import matplotlib.pyplot as plt
import imageio
from io import BytesIO
import numpy as np



def figures_to_gif(figures, output_path='animation.gif', duration=0.2):

    frames = []

    for fig in figures:
        buf = BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        frames.append(imageio.v3.imread(buf))  # or imageio.imread(buf) for older versions
        buf.close()
        plt.close(fig)  # Close to free memory

    imageio.mimsave(output_path, frames, duration=duration)