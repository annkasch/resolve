import numpy as np
np.random.seed(42)
from emukit.multi_fidelity.convert_lists_to_array import convert_x_list_to_array
from resolve.utilities import ModelVisualizer

class GPMultiFidelityVisualizer(ModelVisualizer):
    def __init__(self, mf_model, fidelities, parameters):
        super().__init__(fidelities, parameters)
        self.mf_model = mf_model

    # Drawings of the aquisition function
    def draw_acquisition_func(self, fig, us_acquisition, x_next=np.array([]), outname='model_acquisition.png'):
        SPLIT = 50
        ax2 = fig.axes

        for i, p in enumerate(self.parameters):
            ax2[i].set_title(f"Projected acquisition function - {p}")
            x_plot = [self.x_fixed[:] for _ in range(SPLIT)]
            x_tmp = np.linspace(self.parameters[p][0], self.parameters[p][1], SPLIT)
            for k in range(SPLIT):
                x_plot[k][i] = x_tmp[k]
            x_plot = np.atleast_2d(x_plot)
            X_plot = convert_x_list_to_array([x_plot, x_plot])
            
            acq = us_acquisition.evaluate(X_plot[SPLIT:])
            try:
                color = next(ax2[i].get_prop_cycle())["color"]
            except AttributeError:
                color = "blue"  # Fallback color if cycle is unavailable
            
            ax2[i].plot(x_tmp, acq / acq.max(), color=color)
            
            acq = us_acquisition.evaluate(X_plot[:SPLIT])
            ax2[i].plot(x_tmp, acq / acq.max(), color=color, linestyle="--")
            
            if x_next.any():
                ax2[i].axvline(x_next[0, i], color="red", label="x_next", linestyle="--")
                ax2[i].text(
                    x_next[0, i] + 0.5, 0.95,
                    f"x = {round(x_next[0, i], 1)}",
                    color="red", fontsize=8
                )
            
            ax2[i].set_xlabel(p)
            ax2[i].set_ylabel(r"$\mathcal{I}(x)$")
        fig.savefig(outname,dpi=300, bbox_inches='tight')
        return fig
    
    def get_model_prediction(self,x_data):
        y_pred = []
        sigma1 = []
        sigma2 = []
        sigma3 = []
        for f in range(self.nfidelities):

            if self.mf_model.normalizer.scaler_x != None:
                x_test_tmp = self.mf_model.normalizer.transform_x(x_data[f])
            else:
                x_test_tmp = np.atleast_2d(x_data[f])
            
            x_test_tmp = np.atleast_2d(x_test_tmp)

            mfsm_model_mean = np.empty(shape=[0, 0])
            mfsm_model_var = np.empty(shape=[0, 0])

            for i in range(len(x_test_tmp)):

                    SPLIT = 1
                    x_plot = []
                    for j in range(self.nfidelities):
                        x_plot.append((np.atleast_2d(x_test_tmp[i])))
                    X_plot = convert_x_list_to_array(x_plot)

                    mean_mf_model, var_mf_model = self.mf_model.model.predict(X_plot[f*SPLIT:(f+1)*SPLIT])
                    mfsm_model_mean=np.append(mfsm_model_mean,mean_mf_model[0,0])
                    mfsm_model_var=np.append(mfsm_model_var,var_mf_model[0,0])
            
            if self.mf_model.normalizer.scaler_x != None:
                mfsm_model_mean = self.mf_model.normalizer.inverse_transform_y(mfsm_model_mean).flatten().tolist()
                mfsm_model_var = self.mf_model.normalizer.inverse_transform_variance(mfsm_model_var)
            mfsm_model_std = np.sqrt(mfsm_model_var)

            y_pred.append(mfsm_model_mean)
            for k, sigma_list in zip([1, 2, 3], [sigma1, sigma2, sigma3]):
                sigma_list.append([
                    mfsm_model_mean - k * mfsm_model_std,
                    mfsm_model_mean + k * mfsm_model_std
                ])


        return y_pred, sigma1, sigma2, sigma3