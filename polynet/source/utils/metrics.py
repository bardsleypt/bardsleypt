import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import norm
# TODO: ensure correct/robust metrics from source material


class Metrics(object):
    # Return probit and inv_probit functions for matplotlib scalings
    @staticmethod 
    def probit(x):
        return norm.ppf(np.clip(x, 1e-10, 1 - 1e-10))

    @staticmethod
    def inv_probit(x):
        return norm.cdf(x)

    @staticmethod
    def ecdf(data):
        x = np.sort(data)
        x = x[~np.isnan(x)]
        n = len(x)
        f = np.arange(1, n + 1) / n
        return f, x

    @staticmethod
    def get_eer(scores, labels, pos_label=None, remove_nan=False):
        # Cast to numpy array, collect labeled scores, return NaNs if < 2 classes passed
        scores, labels = np.array(scores), np.array(labels)
        if len(np.unique(labels)) < 2:
            return np.nan, np.nan
        fp, x, fn, _ = Metrics.get_score_dists(scores, labels, pos_label=pos_label,
                                               remove_nan=remove_nan, interp=True)

        # Find zero-crossing of FPR - FNR, interpolate to find eer_thresh, then eer
        fpfn = fp - fn
        eer_interp = interp1d(fpfn, x, kind='linear')
        eer_thresh = eer_interp(0)[()]
        fp_interp = interp1d(x, fp, kind='linear')
        eer = fp_interp(eer_thresh)[()]
        return eer, eer_thresh

    @staticmethod
    def get_thresh(*args, **kwargs):
        return Metrics.get_threshold(*args, **kwargs)

    @staticmethod
    def get_threshold(scores, labels, mode='frr', target=0.005, pos_label=None,
                      remove_nan=False):
        scores, labels = np.array(scores), np.array(labels)
        fp, xp, fn, xn = Metrics.get_score_dists(scores, labels, pos_label=pos_label,
                                                 remove_nan=remove_nan)
        if mode == 'far':
            f, x = fn, xn
        elif mode == 'frr':
            f, x = fp, xp
        else:
            raise Exception('Unsupported threshold mode')
        x_interp = interp1d(f, x, kind='linear')
        threshold = x_interp(target)[()]
        return threshold

    @staticmethod
    def get_fpr(scores, labels, thresh=0., pos_label=None, remove_nan=False):
        # Cast to arrays
        scores, labels = np.array(scores), np.array(labels)

        # Return NaN if no 'negative' labels
        if np.sum(labels != pos_label) == 0:
            return np.nan, thresh

        # Get negative-label score distribution
        _, _, fn, xn = Metrics.get_score_dists(scores[labels != pos_label],
                                               labels[labels != pos_label],
                                               pos_label=pos_label, remove_nan=remove_nan)

        # Interpolate to find rate
        fn_interp = interp1d(xn, fn, kind='linear', fill_value='extrapolate')
        fpr = np.clip(fn_interp(thresh), 0., 1.)
        return fpr, thresh

    @staticmethod
    def get_fnr(scores, labels, thresh=0., pos_label=None, remove_nan=False):
        # Cast to arrays
        scores, labels = np.array(scores), np.array(labels)

        # Return NaN if no 'positive' labels
        if np.sum(labels == pos_label) == 0:
            return np.nan, thresh

        # Get positive-label score distribution
        fp, xp, _, _ = Metrics.get_score_dists(scores[labels == pos_label],
                                               labels[labels == pos_label],
                                               pos_label=pos_label, remove_nan=remove_nan)

        # Interpolate to find rate
        fp_interp = interp1d(xp, fp, kind='linear', fill_value='extrapolate')
        fnr = np.clip(fp_interp(thresh), 0., 1.)
        return fnr, thresh

    @staticmethod
    def get_score_dists(scores, labels, pos_label=None, remove_nan=False, interp=False):
        # Cast to arrays
        p_scores, n_scores = np.array(scores[labels == pos_label]), \
            np.array(scores[labels != pos_label])

        # Get CDFs, reverse the negative-labeled CDF
        fp, xp = Metrics.ecdf(p_scores)
        fn, xn = Metrics.ecdf(n_scores)
        fn = 1. - fn

        # Adjust maximal rate of CDFs if retaining NaNs in denominator of rate
        # computations, e.g.,:
        #       - retain NaNs FRR = (# pos class rejected) / (# pos class files)
        #           - maximal FRR is (# non-nan pos class files) / (# pos class files)
        #       - remove NaNs FRR = (# pos class rejected) / (# non-nan pos class files)
        #           - maximal FRR is 1 = (# non-nan pos class files) / (# non-nan pos cls)
        if len(p_scores) > 0:
            max_perc_p = 1. if remove_nan else np.sum(~np.isnan(p_scores)) / len(p_scores)
            fp = max_perc_p * fp
        if len(n_scores) > 0:
            max_perc_n = 1. if remove_nan else np.sum(~np.isnan(n_scores)) / len(n_scores)
            fn = max_perc_n * fn

        # Interpolate fp, fn to common scores if required
        if interp:
            if any([len(scores) == 0 for scores in [p_scores, n_scores]]):
                # Cannot interpolate to common scores since p or n scores are missing,
                return fp, xp, fn, xn
            x = np.sort(np.concatenate([xp, xn]))
            fp_interp = interp1d(xp, fp, kind='linear', fill_value='extrapolate')(x)
            fp = np.clip(fp_interp, 0., fp[-1])
            fn_interp = interp1d(xn, fn, kind='linear', fill_value='extrapolate')(x)
            fn = np.clip(fn_interp, 0., fn[0])
            xp, xn = x, x
        return fp, xp, fn, xn

    @staticmethod
    def get_det_curve(scores, labels, pos_label=None, remove_nan=False, num_interp=100):
        # Cast to numpy arrays, setup return dictionary
        scores, labels = np.array(scores), np.array(labels)
        ret_dict = dict.fromkeys(['scores', 'far', 'frr', 'probit', 'inv_probit'])

        # Return NaNs if missing p_socres or n_scores
        if len(np.unique(labels)) < 2:
            return ret_dict

        # Get score distirbutions, interp=True for common score points
        fp, x, fn, _ = Metrics.get_score_dists(scores, labels, pos_label=pos_label,
                                               remove_nan=remove_nan, interp=True)

        # Remove extremal values to avoid infinities in DET curve
        idx = np.where((fp > 0.) & (fp < fp[-1]) &
                       (fn > 0.) & (fn < fn[0]))[0]
        ret_dict = {'scores': x[idx], 'far': fn[idx], 'frr': fp[idx],
                    'probit': Metrics.probit, 'inv_probit': Metrics.inv_probit}

        # Interpolate DET curve on probit scale to 'downsample' resulting curves
        if num_interp is not None:
            far_p, frr_p = [Metrics.probit(ret_dict[x]) for x in ['far', 'frr']]
            far_interp = np.linspace(far_p[0], far_p[-1], num_interp)
            frr_interp = interp1d(far_p, frr_p, kind='linear')(far_interp)
            scores_interp = interp1d(far_p, ret_dict['scores'], kind='linear')(far_interp)
            ret_dict.update({'far': Metrics.inv_probit(far_interp),
                             'frr': Metrics.inv_probit(frr_interp),
                             'scores': scores_interp})

        return ret_dict

    @staticmethod
    def get_prec_rec(preds, labels, mode='macro'):
        def macro_metrics(preds_, labels_):
            # Compute precision, recall per label
            prec_, rec_ = {}, {}
            for lab in np.unique(labels_):
                pos_idx = (labels_ == lab)
                prec_[lab] = np.nan_to_num(np.sum(preds_[pos_idx] == lab) /
                                           np.sum(preds_ == lab), nan=0.)
                rec_[lab] = np.nan_to_num(np.sum(preds_[pos_idx] == lab) /
                                          np.sum(pos_idx), nan=0.)

            prec_ = np.mean([x for x in prec_.items()])
            rec_ = np.mean([x for x in rec_.items()])
            return prec_, rec_

        preds, labels = np.array(preds), np.array(labels)
        if preds.ndim > 1:
            preds = np.argmax(preds, axis=1)
        if labels.ndim > 1:
            labels = np.argmax(labels, axis=1)

        prec, rec = np.nan, np.nan
        if mode == 'macro':
            prec, rec = macro_metrics(preds, labels)
        elif mode == 'micro':
            # TODO: Write micro_metrics, replace call here
            prec, rec = macro_metrics(preds, labels)
        return prec, rec
