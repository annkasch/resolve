import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

def place_text_corner(ax, text, corner="bottom left", offset=0.03, **kwargs):
    positions = {
        "bottom left":  (offset, offset),
        "bottom right": (1 - offset, offset),
        "top left":     (offset, 1 - offset),
        "top right":    (1 - offset, 1 - offset),
    }
    x, y = positions.get(corner.lower(), (offset, offset))
    ax.text(x, y, text, transform=ax.transAxes,
            ha='left' if 'left' in corner else 'right',
            va='bottom' if 'bottom' in corner else 'top',
            **kwargs)

class ModelVisualizer:
    def __init__(self, fidelities, parameters):
        self.parameters = parameters
        self.x_min = np.array([parameters[k][0] for k in parameters])
        self.x_max = np.array([parameters[k][1] for k in parameters])
        self.fidelities = fidelities
        self.nfidelities = len(fidelities)
        self.feature_labels = list(map(str, parameters.keys()))
        
    def get_marginalized_single_draw(self, x_data, y_data, keep_axis, grid_steps=25):
        """
        Marginalizes predictions over all but one feature using random sampling when only one 
        prediction is available per sample (y_hf has shape (n_samples, 1)).
        """
        x_keep = x_data[:, keep_axis]

        # 4. Define bins along the kept axis in the original scale.
        bin_edges = np.linspace(self.x_min[keep_axis], self.x_max[keep_axis], grid_steps + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

                # 5. For each bin, compute the median and 1σ percentiles (16th and 84th) from the samples in the bin.
        medians = np.empty(grid_steps)
        one_sigma_lower_vals = np.empty(grid_steps)
        one_sigma_upper_vals = np.empty(grid_steps)
        two_sigma_lower_vals = np.empty(grid_steps)
        two_sigma_upper_vals = np.empty(grid_steps)
        three_sigma_lower_vals = np.empty(grid_steps)
        three_sigma_upper_vals = np.empty(grid_steps)
        for i in range(grid_steps):
            # Use a half-open interval except for the last bin.
            if i < grid_steps - 1:
                mask = (x_keep >= bin_edges[i]) & (x_keep < bin_edges[i+1])
            else:
                mask = (x_keep >= bin_edges[i]) & (x_keep <= bin_edges[i+1])
            if np.sum(mask) > 0:
                bin_values = y_data[mask]
                medians[i] = np.median(bin_values)
                one_sigma_lower_vals[i] = np.percentile(bin_values, 16)
                one_sigma_upper_vals[i] = np.percentile(bin_values, 84)
                two_sigma_lower_vals[i] = np.percentile(bin_values, 2.5)
                two_sigma_upper_vals[i] = np.percentile(bin_values, 97.5)
                three_sigma_lower_vals[i] = np.percentile(bin_values, 0.5)
                three_sigma_upper_vals[i] = np.percentile(bin_values, 99.5)
            else:
                medians[i] = np.nan
                one_sigma_lower_vals[i] = np.nan
                one_sigma_upper_vals[i] = np.nan
                two_sigma_lower_vals[i] = np.nan
                two_sigma_upper_vals[i] = np.nan
                three_sigma_lower_vals[i] = np.nan
                three_sigma_upper_vals[i] = np.nan
        # Compute errors for plotting (errorbars represent the distance from the median to the percentiles)
        one_sigma_lower_error = medians - one_sigma_lower_vals
        one_sigma_upper_error = one_sigma_upper_vals - medians
        two_sigma_lower_error = medians - two_sigma_lower_vals
        two_sigma_upper_error = two_sigma_upper_vals - medians
        three_sigma_lower_error = medians - three_sigma_lower_vals
        three_sigma_upper_error = three_sigma_upper_vals - medians
        
        return bin_centers, medians, [one_sigma_lower_error, one_sigma_upper_error], [two_sigma_lower_error, two_sigma_upper_error] ,[three_sigma_lower_error, three_sigma_upper_error]
    

    def get_model_prediction(self, x_data):
        pass

    def plot_projection(self,x_data, y_data, with_prediction=False, grid_steps=20):
                nrows = self.nfidelities
                ncols = len(self.x_max)
                fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10 * ncols, 4 * nrows), sharex='col', sharey='row',squeeze=False)
                if with_prediction:
                    y_prediction,_,_,_ = self.get_model_prediction(x_data)
                for f in range(nrows):

                    for keep_axis in range(ncols):
                        ax = axes[f][keep_axis]
                        if with_prediction:
                            x, y_mean, y_1sigma, y_2sigma, y_3sigma = self.get_marginalized_single_draw(x_data[f], y_prediction[f], keep_axis=keep_axis, grid_steps=grid_steps)
                            
                            ax.fill_between(
                                x, y_mean-y_3sigma[0], y_mean+y_3sigma[1],
                                color="coral", alpha=0.2, label=r'$\pm 3\sigma$'
                            )
                            ax.fill_between(
                                x, y_mean-y_2sigma[0], y_mean+y_2sigma[1],
                                color="yellow", alpha=0.2, label=r'$\pm 2\sigma$'
                            )
                            
                            ax.fill_between(
                                x, y_mean-y_1sigma[0], y_mean+y_1sigma[1],
                                color="green", alpha=0.2, label=r'$\pm 1\sigma$'
                            )
                            ax.plot(x, y_mean, color="gray", label="Model")

                        x, y, [y_low, y_high],_,_ = self.get_marginalized_single_draw(x_data[f], y_data[f], keep_axis=keep_axis, grid_steps=grid_steps)

                        ax.errorbar(
                            x, y,
                            yerr=[y_low, y_high],
                            fmt='o',                   # 'o' for circular markers
                            color='black',             # marker & line color
                            ecolor='black',            # error bar color
                            elinewidth=0.5,               # error bar line width
                            capsize=3,                  # length of the error bar caps
                            markersize=4,               # size of scatter points
                            label="Data"
                        )



                        ax.set_xlabel(f'{list(self.parameters.keys())[keep_axis]}',fontsize=16)
                        str_tmp = f"{self.fidelities[f]}"
                        ax.set_ylabel('Marginalized predicted $\epsilon^{('+str_tmp+')}$', fontsize=16)


                n_rows = axes.shape[0]
                for i in range(n_rows - 1):
                    for ax in axes[i, :]:
                        ax.set_xlabel("")
                
                for i in range(n_rows):
                    for ax in axes[i, 1:]:
                        ax.set_ylabel("")

                # Create custom legend handles
                
                legend_elements = [
                    Line2D([0], [0], marker='.', color='black', linestyle='None', label='Data')
                ]
                if with_prediction:
                    legend_elements.extend(
                        [Line2D([0], [0], color='black', lw=2, label='Model prediction'),
                        mpatches.Patch(color='green', alpha=0.2, label=r'$\pm 1\sigma$'),
                        mpatches.Patch(color='yellow', alpha=0.2, label=r'$\pm 2\sigma$'),
                        mpatches.Patch(color='coral', alpha=0.2, label=r'$\pm 3\sigma$')]
                    )

                fig.legend(handles=legend_elements, loc='upper center', ncol=len(legend_elements), fontsize=16, frameon=False, bbox_to_anchor=(0.5, 1.05))
                plt.tight_layout(rect=[0, 0.05, 1, 1])  # Leave space at bottom for legend
                return fig

    def validate_mse(self, x_data, y_data):
        """
        Validate the mean squared error (MSE) of the model.
        Parameters:
        - x_data (ndarray): Input data for prediction.
        - y_data (ndarray): True target values.

        Returns:
        - list: Mean Squared Error
        """
        mse = []
        y_mean,_,_,_ = self.get_model_prediction(x_data)

        for f in range(self.nfidelities):
            mse.append(np.mean((y_data[f] - y_mean[f]) ** 2))

        return mse
    
    def validate_coverage(self, x_data, y_data):
        """
        Validate the coverage of the model for 1, 2, and 3 sigma intervals.

        Parameters:
        - y_data (ndarray): True high-fidelity target values for validation.
        - y_hf_pred_samples (ndarray): Posterior predictive samples for high-fidelity predictions.

        Returns:
        - dict: Percentages of validation data within 1, 2, and 3 sigma intervals.
        """
        coverage={}
        _, sigma1, sigma2, sigma3 = self.get_model_prediction(x_data)
        for ix in range(len(x_data)):

            y_data_tmp = y_data[ix]
            counters = {1: 0, 2: 0, 3: 0}

            # Calculate percentile intervals for the posterior samples
            percentiles = {
                1: (sigma1[ix][0], sigma1[ix][1]),
                2: (sigma2[ix][0], sigma2[ix][1]),
                3: (sigma3[ix][0], sigma3[ix][1]),
            }

            # Count the number of y_data points within each interval
            for i, y in enumerate(y_data_tmp):
                for sigma in [1, 2, 3]:
                    low, high = percentiles[sigma]
                    if low[i] <= y <= high[i]:
                        counters[sigma] += 1

            # Calculate percentages
            coverage[self.fidelities[ix]]={sigma: (counters[sigma] / len(y_data_tmp)) * 100 for sigma in [1, 2, 3]}
        return coverage
    
    def plot_validation(self, x_data, y_data):
        """
        Plot the validation data with the uncertainty prediction bands.
        Parameters:
        - x_data (ndarray): Input data (e.g., validation or test set).
        - y_data (ndarray): True high-fidelity target values for validation.
        """
        if len(x_data) != self.nfidelities:
            print(f"ERROR: Expected data for {self.nfidelities} fidelities, but got {len(x_data)}.")
            return

        mse = self.validate_mse(x_data,y_data)
        coverage = self.validate_coverage(x_data,y_data)

        nrows = self.nfidelities
        ncols = 1
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12 * ncols, 3 * nrows), squeeze=False)
        _, sigma1, sigma2, sigma3 = self.get_model_prediction(x_data)

        for f in range(self.nfidelities):
            ax = axes[f][0]

            sample_numbers = np.arange(len(y_data[f]))

            ax.fill_between(
                sample_numbers,
                sigma3[f][0],
                sigma3[f][1],
                color="coral", alpha=0.2, label=r'$\pm 3\sigma$'
            )
            ax.fill_between(
                sample_numbers,
                sigma2[f][0],
                sigma2[f][1],
                color="yellow", alpha=0.2, label=r'$\pm 2\sigma$'
            )
            ax.fill_between(
                sample_numbers,
                sigma1[f][0],
                sigma1[f][1],
                color="green", alpha=0.2, label=r'$\pm 1\sigma$'
            )
            ax.scatter(sample_numbers, y_data[f], marker='.',s=5, color="black", label=f"{self.fidelities[f]} Validation Data")

            ax.set_xlabel(f"Simulation Trial Number")
            ax.set_ylabel(r"Predicted $\epsilon^{("+f"{self.fidelities[f]}"+")}$")
            text = f"MSE: {mse[f]:.7f} $\pm1\sigma$: {coverage[self.fidelities[f]][1]:.1f}%  $\pm3\sigma$: {coverage[self.fidelities[f]][2]:.1f}%  $\pm3\sigma$: {coverage[self.fidelities[f]][3]:.1f}%"
            place_text_corner(ax, text, fontsize=8, bbox=dict(edgecolor='gray', facecolor='none', linewidth=0.5))
            
        legend_elements = [
            Line2D([0], [0], marker='.', color='black', linestyle='None', label='Data'),
            Line2D([0], [0], marker='.', color='white', linestyle='None', label='Model prediction'),
            mpatches.Patch(color='green', alpha=0.2, label=r'$\pm 1\sigma$'),
            mpatches.Patch(color='yellow', alpha=0.2, label=r'$\pm 2\sigma$'),
            mpatches.Patch(color='coral', alpha=0.2, label=r'$\pm 3\sigma$')
        ]
        fig.legend(handles=legend_elements, loc='upper center', ncol=len(legend_elements), fontsize='medium', frameon=False, bbox_to_anchor=(0.5, 1.05))
        plt.tight_layout()
        return fig
    