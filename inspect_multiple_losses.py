from helper_functions import *
# from settings import *
import pickle
import matplotlib.pyplot as plt
import math


class LossInspector:
    def __init__(self, s, h):
        self.s = s
        self.h = h

    def inspect_loss(self):
        # model_names = ['gs_art_fraction_2/1', 'gs_art_fraction_2/2', 'gs_art_fraction_2/3', 'gs_art_fraction_2/4']
        model_names = ['la_seg_new_data', 'la_challenge_data']
        colors = ['b', 'r', 'g', 'y']
        legend_parameter = 'ART_FRACTION'
        legend_parameter_name = 'Artificial data fraction'

        plt.figure()
        legend = []

        w = 600
        orig_lw = 1
        smooth_lw = 2

        show_non_smooth = False

        for j in range(len(model_names)):
            model_name = model_names[j]
            color = colors[j]
            log_path = self.h.getLogPath(model_name)
            log = pickle.load(open(log_path, "rb"))

            legend.append('{} = {}'.format(legend_parameter_name, eval("log['settings'].{}".format(legend_parameter))))

            # w = int(round(len(log['training']['loss'])/10))

            m = len(log['training'].keys())

            if show_non_smooth:
                cnt = 1

                for i in log['training']:
                    plt.subplot(2, m, cnt)
                    plt.plot(log['training'][i], lw=orig_lw, alpha=.3, color=color)

                    plt.title('train: ' + i, fontsize=8)

                    plt.subplot(2, m, m + cnt)
                    plt.plot(log['validation'][i], lw=orig_lw, alpha=.3, color=color)
                    plt.title('val: ' + i, fontsize=8)

                    cnt += 1

            cnt = 1
            for i in log['training']:
                plt.subplot(2, m, cnt)
                plt.plot(self.h.smooth(log['training'][i], w), lw=smooth_lw, color=color)

                plt.title('train: ' + i, fontsize=8)

                plt.subplot(2, m, m + cnt)
                plt.plot(self.h.smooth(log['validation'][i], w), lw=smooth_lw, color=color)
                plt.title('val: ' + i, fontsize=8)

                cnt += 1
        plt.legend(legend)
        plt.show()


if __name__ == "__main__":
    s = Settings()
    h = Helper(s)
    log_inspector = LossInspector(s, h)
    log_inspector.inspect_loss()
