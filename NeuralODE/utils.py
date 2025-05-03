import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
import matplotlib.animation as animation
from tqdm import tqdm

def create_learning_gif(result, model_names, log_every, exp_dir, gif_filename="prediction_progress.gif"):

    # convert tensors to np.arrays
    ground_truth = np.array(result['ground_truth'])
    for mname in model_names:
        result[mname] = [np.array(r) for r in result[mname]]

    # quadratic plot, takes the first n square-rootable elements of test batch
    n_rows_cols = int(np.sqrt(ground_truth.shape[0]))
    fig, axs = plt.subplots(nrows=n_rows_cols, ncols=n_rows_cols, figsize=(15, 15), sharex=True, sharey=True)
    axs = np.atleast_2d(axs)

    # axis limits set by ground truth of larges sample in test batch, result (x,y)
    min_value = np.min(ground_truth, axis=(0,1))-0.2
    max_value = np.max(ground_truth, axis=(0,1))+0.2
    writer = PillowWriter(fps=5)
    
    num_frames = len(result[model_names[0]])
    
    with writer.saving(fig, os.path.join(exp_dir, gif_filename), dpi=80):
        with tqdm(total=num_frames) as pbar:
            
            for i in range(num_frames):

                pbar.set_description(f'processing frame {i}/{num_frames} (iteration {log_every * i}) ...')

                for j, ax in enumerate(axs.flatten()):
                    ax.clear()
                    for mname in model_names:
                        ax.plot(*(np.array(result[mname][i][j, :, :]).T), label=mname)
                    ax.plot(*(np.array(ground_truth[j, :, :]).T), label='GT')
                    ax.set_xlim(min_value[0], max_value[0])
                    ax.set_ylim(min_value[1], max_value[1])

                # Figure-wide legend (only using the one of first figure)
                handles, labels = axs[0,0].get_legend_handles_labels()
                fig.legend(handles, labels, loc='upper right', ncol=2, fontsize=28)
                fig.suptitle(f"Iteration {(i + 1)*log_every}", fontsize=30, x=0.1, ha='left')
                writer.grab_frame()

                pbar.update()

    plt.close(fig)
    print(f"GIF saved to {os.path.join(exp_dir, gif_filename)}")