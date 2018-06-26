from helper_functions import *
# from settings import *
import pickle
import matplotlib.pyplot as plt
import math


class LogInspector:
    def __init__(self, s, h):
        self.s = s
        self.h = h

    def inspect_log(self):
        log_path = self.h.getLogPath(self.s.MODEL_NAME)
        log = pickle.load(open(log_path, "rb"))

        print(log)

        logs_to_output = ['stopped_early', 'lowest_val_loss', 'lowest_val_loss_i']
        for log_name in logs_to_output:
            if log_name in log:
                print('{:>27} = {}'.format(log_name, log[log_name]))
            else:
                print('{:>27} = absent in this log file'.format(log_name))

        lowest_training_loss = min(log['training']['loss']) if len(log['training']['loss']) > 0 else 'absent'

        print('{:>27} = {}'.format('lowest training loss', lowest_training_loss))

        lowest_training_loss_i = np.argmin(log['training']['loss']) if len(log['training']['loss']) > 0 else 'absent'
        print('{:>27} = {}'.format('lowest training loss i', lowest_training_loss_i))

        settings_to_output = ['MODEL_NAME', 'FN_CLASS_WEIGHT', 'UNET_DEPTH', 'LEARNING_RATE', 'PATCH_SIZE', 'DROPOUT',
                              'FEATURE_MAP_INC_RATE', 'LOSS_FUNCTION', 'BATCH_SIZE', 'NR_AUG', 'NR_DIM', 'ART_FRACTION',
                              'POS_NEG_PATCH_PROP', 'PATIENTCE_ES', 'USE_ANY_SCAR_AUX', 'MAIN_OUTPUT_LOSS_WEIGHT',
                              'AUX_OUTPUT_LOSS_WEIGHT', 'NR_CONV_PER_CONV_BLOCK', 'START_CH']
        for name in settings_to_output:
            try:
                expr = "log['settings'].{}".format(name)
                print('s.{:>25} = {}'.format(name, eval(expr)))
            except:
                print('s.{:>25} = absent in this log file'.format(name))

        # w = 50
        w = int(round(len(log['training']['loss'])/20))
        orig_lw = 1
        smooth_lw = 2

        plt.figure()

        m = len(log['training'].keys())
        cnt = 1
        for i in log['training']:
            plt.subplot(2, m, cnt)
            # plt.plot(log['training'][i], lw=orig_lw, alpha=.3)
            plt.plot(self.h.smooth(log['training'][i], w), lw=smooth_lw)

            # plt.plot(np.log10(log['training'][i]), lw=orig_lw, alpha=.3)
            # plt.plot(np.log10(self.h.smooth(log['training'][i], w)), lw=smooth_lw)
            plt.title('train: ' + i, fontsize=8)

            plt.subplot(2, m, m + cnt)
            # plt.plot(log['validation'][i], lw=orig_lw, alpha=.3)
            plt.plot(self.h.smooth(log['validation'][i], w), lw=smooth_lw)

            # plt.plot(np.log10(log['validation'][i]), lw=orig_lw, alpha=.3)
            # plt.plot(np.log10(self.h.smooth(log['validation'][i], w)), lw=smooth_lw)
            plt.title('val: ' + i, fontsize=8)

            cnt += 1
        plt.show()


if __name__ == "__main__":
    s = Settings()
    h = Helper(s)
    log_inspector = LogInspector(s, h)
    log_inspector.inspect_log()
