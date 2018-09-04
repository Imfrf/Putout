# Putout
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt
import TrackCenters
import plot_event3
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from TrackCenters import *
import numpy as np
from scipy.spatial.distance import cdist

XY_NAME = ['xe0', 'ye0']


# perform Hough Transform
class HoughTransform(object):

    def __init__(self, data_sig, data_bak, y_pre, xy_name=XY_NAME):
        hf = TrackCenters()
        r_max = hf.r_max
        r_min = hf.r_min
        trk_rho_sgma = hf.trk_rho_sgma

        xy_sig = data_sig[xy_name]
        xy_bak = data_bak[xy_name]
        xy_hits = np.concatenate((xy_sig, xy_bak), axis=0)

        n_of_points = np.sum(hf.n_by_layer)

        dist = np.zeros([len(xy_hits), n_of_points])
        result = np.zeros([len(xy_hits), n_of_points])
        weight = np.transpose([y_pre])
        vt_points = np.zeros(n_of_points)

        # vote on track centers/points
        dist = cdist(xy_hits, hf.xy_points)
        result = np.where(dist <= r_max + trk_rho_sgma, dist, 0)
        result = np.where(result >= r_min - trk_rho_sgma, 1, 0)
        vote_table = result * weight

        self.vt_points_sig = np.sum(result[:len(data_sig)], axis=0)  # ##################
        max_vpon_sig = np.amax(self.vt_points_sig)
        min_vpon_sig = np.amin(self.vt_points_sig)
        self.vt_points_sig = (self.vt_points_sig - min_vpon_sig) / (max_vpon_sig - min_vpon_sig)  # normalized

        vt_points = vote_table.sum(axis=0)
        max_vpon = np.amax(vt_points)
        min_vpon = np.amin(vt_points)
        # print(max_vpon, min_vpon)
        vt_points = (vt_points - min_vpon) / (max_vpon - min_vpon) * 15
        self.vt_points = vt_points / 15  # normalized

        # vote on signals and backgrounds
        wet_points = np.exp(vt_points)
        self.vt_hits = np.sum(result * wet_points, axis=1)
        vt_max = np.amax(self.vt_hits)
        vt_min = np.amin(self.vt_hits)
        self.vt_hits = (self.vt_hits - vt_min) / (vt_max - vt_min)  # normalized
        self.vt_sigs = self.vt_hits[:len(data_sig)]
        self.vt_baks = self.vt_hits[len(data_sig):]

def main():
    seed = 10

    signals = pd.read_csv("/home/wangkaipu/IHEP/data_new/signals.csv")
    backgrounds = pd.read_csv("/home/wangkaipu/IHEP/data_new/backgrounds.csv")
    train_sig = signals[:5000]
    train_bak = backgrounds[:4500]
    test_sig = signals[5000:7500]
    test_bak = backgrounds[4500:7000]

    train = pd.concat([train_sig, train_bak])
    test = pd.concat([test_sig, test_bak])
    # jiance

    x_columns = [x for x in train if x in ['r', 'phi', 'ldetmt0', 'rdetmt0', 'mdetmt0', 'le', 'me', 're', 'layer']]
    X = train[x_columns]
    x = train['isSignal']
    z_columns = [x for x in test if x in ['r', 'phi', 'ldetmt0', 'rdetmt0', 'mdetmt0', 'le', 'me', 're', 'layer']]
    Z = test[z_columns]
    z = test['isSignal']

    local_gbdt = GradientBoostingClassifier(random_state=30, learning_rate=0.1, n_estimators=150, min_samples_leaf=20,
                                            max_features='sqrt', subsample=0.9, max_depth=5, min_samples_split=550)

    local_gbdt.fit(X, x)
    z_pre = local_gbdt.predict(Z)
    z_pre_prob = local_gbdt.predict_proba(Z)[:, 1]
    x_pre = local_gbdt.predict(X)
    x_pre_prob = local_gbdt.predict_proba(X)[:, 1]

    weit_train_sig = x_pre_prob[:5000]
    weit_test_sig = z_pre_prob[:2500]
    weit_train_bak = x_pre_prob[5000:9500]
    weit_test_bak = z_pre_prob[2500:5000]
    weit_sig = np.hstack((weit_train_sig, weit_test_sig))
    weit_bak = np.hstack((weit_train_bak, weit_test_bak))
    weit = np.hstack((weit_sig, weit_bak))
    signal = signals[:7500]
    background = backgrounds[:7000]
    hft = HoughTransform(signal, background, weit)
    vt_sig = hft.vt_sigs
    vt_bak = hft.vt_baks
    vt_result = hft.vt_points
    vresult_sig = hft.vt_points_sig
    vt_train_sig = vt_sig[:5000]
    vt_train_bak = vt_bak[:4500]
    vt_test_sig = vt_sig[5000:7500]
    vt_test_bak = vt_bak[4500:7000]
    vt_train = np.hstack((vt_train_sig, vt_train_bak))
    vt_test = np.hstack((vt_test_sig, vt_test_bak))
    local_gbdt1 = GradientBoostingClassifier(random_state=30, learning_rate=0.1, n_estimators=150, min_samples_leaf=20,
                                             max_features='sqrt', subsample=0.9, max_depth=5, min_samples_split=550)

    local_gbdt1.fit(X, x, sample_weight=vt_train)
    z_pre_hough = local_gbdt1.predict(Z)
    z_pre_prob_hough = local_gbdt1.predict_proba(Z)[:, 1]
    x_pre_hough = local_gbdt1.predict(X)
    x_pre_prob_hough = local_gbdt1.predict_proba(X)[:, 1]

    print("test_Accuracy: %.4g" % metrics.accuracy_score(z, z_pre))
    print("test_Pro_Accuracy: %.4g" % metrics.roc_auc_score(z, z_pre_prob))
    print("train_Accuracy:%.4g" % metrics.accuracy_score(x, x_pre))
    print("train_Pro_Accuracy:%.4g" % metrics.roc_auc_score(x, x_pre_prob))

    print("The results of Houghtrsnsform:")
    print("test_Accuracy:%.4g" % metrics.accuracy_score(z, z_pre_hough))
    print("test_Pro_Accuracy:%.4g" % metrics.roc_auc_score(z, z_pre_prob_hough))
    print("test_Pro_Accuracy_by_sample_weight:%.4g" % metrics.roc_auc_score(z, z_pre_prob_hough))

    # plot fpr and tpr
    fpr, tpr, thresholds = roc_curve(z, z_pre_prob)
    fpr_hough, tpr_hough, thresholds_hough = roc_curve(z, z_pre_prob_hough)
    roc_auc = auc(fpr, tpr)
    roc_auc_hough = auc(fpr_hough, tpr_hough)

    fig = plt.figure(figsize=(8, 4))
    ax1 = fig.add_subplot(121)
    plt.title('Receiver Operating Characteristic')
    ax1.plot(fpr, tpr, 'b', label='AUC = %0.4f' % roc_auc)
    ax1.legend(loc='lower right')
    ax1.plot([0, 1], [0, 1], 'r--')
    ax1.set_xlim([-0.1, 1.2])
    ax1.set_ylim([-0.1, 1.2])
    ax1.set_ylabel('True Positive Rate')
    ax1.set_xlabel('False Positive Rate')
    ax2 = fig.add_subplot(122)
    plt.title('Receiver Operating Characteristic\n ----by Houghtransform')
    ax2.plot(fpr_hough, tpr_hough, 'g', label='AUC = %0.4f' % roc_auc_hough)
    ax2.legend(loc='lower right')
    ax2.plot([0, 1], [0, 1], 'r--')
    ax2.set_xlim([-0.1, 1.2])
    ax2.set_ylim([-0.1, 1.2])
    ax2.set_ylabel('True Positive Rate')
    ax2.set_xlabel('False Positive Rate')

    plt.show()

    print("x_pre_prob_hough:", x_pre_prob_hough)
    print("z_per_prov_hough:", z_pre_prob_hough)
    print("min_event_id:", min(signal.event_id), "max_event_id:", max(signal.event_id))
    max_event_id_sig = max(signal.event_id)
    min_event_id_sig = min(signal.event_id)

    hough = plot_event3.Houghspace()
    z_pre_prob_hough -= 1

    hough = TrackCenters()

    hft_by_data = HoughTransform(signal, background, y_pre=weit)
    plot_event3.putout(hough=hough, signals=signal, backgrounds=background, vt_bak=hft_by_data.vt_baks,
                       vt_sig=hft_by_data.vt_sigs,
                       vresult=hft_by_data.vt_points, vresult_sig=hft_by_data.vt_points_sig, out=True,
                       trackcenter=False, circlebysig=False, circlebytrackcenter=False,
                       tkctrbywt=True, backgrounds_=False, backgroundsbywt=True, signals_=False, signalsbywt=True,
                       sub_data=True)
    plot_event3.putout(hough=hough, signals=signal, backgrounds=background, vt_sig=x_pre_prob_hough,
                       vt_bak=z_pre_prob_hough,
                       sub_data=False, min_event_id=min_event_id_sig, max_event_id=max_event_id_sig, vresult=vt_result,
                       vresult_sig=vresult_sig, out=True, trackcenter=False, circlebysig=False,
                       circlebytrackcenter=True,
                       tkctrbywt=True, backgrounds_=False, backgroundsbywt=True, signals_=False, signalsbywt=True)
    # para:out function is called or not
if __name__=='__main__':
    main()
