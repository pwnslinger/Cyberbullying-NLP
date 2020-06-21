import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score

class Writer(object):
    def __init__(self, ds_name, y_test, y_pred, clf_method, vec_method):
        self.ds_name = ds_name
        self.y_test = y_test
        self.y_pred = y_pred
        self.clf_method = clf_method
        self.vec_method = vec_method
        self._roc_curve_fname = "%s_%s_ROC.png" % (self.clf_method, self.vec_method)
        self._conf_matrix_fname = "%s_%s_confusion.png" % (self.clf_method, self.vec_method)

    def generate_plots(self):
        dir_path = self._make_dirs()
        self._plot_roc(dir_path)
        self._plot_confusion_matrix(dir_path)

    def classification_report(self, best_estimator):
        results = "%s with %s\n"%(self.clf_method, self.vec_method)
        results += "%s\n" % best_estimator
        results += classification_report(self.y_test, self.y_pred)
        results += "AUC score: %s\n"%roc_auc_score(self.y_test, self.y_pred)
        results += "\n-----------------------------------\n"
        print(results)
        return results

    def _make_dirs(self):
        dir_path = './results/%s' % self.ds_name

        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)

        return dir_path

    def _plot_roc(self, plot_path):
        # calculate auc
        clf_auc = roc_auc_score(self.y_test, self.y_pred)

        # calculate roc curve
        clf_fpr, clf_tpr, _ = roc_curve(self.y_test, self.y_pred)

        # Title
        plt.title('ROC Plot-%s on %s'%(self.clf_method, self.vec_method))

        # Axis labels
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')

        # Show legend
        plt.legend()

        # plot
        plt.plot(clf_fpr, clf_tpr, marker='.', label='%s_%s (AUROC = %0.3f)' %
                (self.clf_method, self.vec_method, clf_auc))

        # Save plot
        plt.savefig(os.path.join(plot_path, self._roc_curve_fname))

        # clear plot
        plt.clf()

    def _plot_confusion_matrix(self, plot_path):
        # calculate confusion matrix
        conf_matrix = confusion_matrix(self.y_test, self.y_pred)
        sns.heatmap(conf_matrix, annot=True)

        plt.title('Confusion matrix Plot-%s on %s'%(self.clf_method, self.vec_method))

        # Axis labels
        plt.xlabel('Predicted')
        plt.ylabel('Actual')

        # Save plot
        plt.savefig(os.path.join(plot_path, self._conf_matrix_fname))

        # clear plot
        plt.clf()
