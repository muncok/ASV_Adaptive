import numpy as np
from spk_model import spk_model

class sv_system():
    def __init__(self, spk_models, config):
        self.spk_models = spk_models
        self.enrolled_spks = [model.name for model in spk_models]
        self.sv_mode = config['sv_mode']
        self.config = config

        self.log_trial_keys = []
        self.log_trial_cfid = []
        self.log_trial_res  = []
        self.log_trial_mod  = []

        self.n_benefits = 0
        self.n_fps = 0
        self.fps_scores = []

    def _get_cfids(self, in_utter):
        cfids = []
        for model in self.spk_models:
            cfid = model.confidence(in_utter)
            cfids.append(cfid)

        return cfids

    def verify_and_enroll(self, key, in_utter):
        verify_result = 0
        enroll_result = -1
        cfids = self._get_cfids(in_utter)
        max_cfid, max_idx = np.max(cfids), np.argmax(cfids)
        selected_model = self.spk_models[max_idx]
        accept_thres = selected_model.accept_thres
        enroll_thres = selected_model.enroll_thres

        if (max_cfid > accept_thres):
            verify_result = 1

        if (max_cfid > enroll_thres) \
                and (self.sv_mode != 'base'):
            enroll_result = selected_model.enroll(key, in_utter, max_cfid)

        return verify_result, enroll_result, max_cfid,

    def verify_and_enroll_neg(self, key, in_utter):
        verify_result = 0
        enroll_result = -1
        cfids = self._get_cfids(in_utter)
        max_cfid, max_idx = np.max(cfids), np.argmax(cfids)
        selected_model = self.spk_models[max_idx]
        accept_thres = selected_model.accept_thres
        enroll_thres = selected_model.enroll_thres
        neg_thres = selected_model.neg_thres
        spk_name = key[:7]

        # supervisely enroll imposters
        # if spk_name not in self.enrolled_spks:
                # self.spk_models.append(
                        # spk_model(self.config,
                            # spk_name, [key], [in_utter]))
                # self.enrolled_spks.append(spk_name)

        # negative enrollment's benefits for reducing fpr
        if len(self.spk_models) > 2:
            if (max_idx != 0)\
                    and (cfids[0] > accept_thres \
                    and spk_name != self.spk_models[0].name):
                self.n_benefits += 1

            if (cfids[0] > accept_thres \
                    and spk_name != self.spk_models[0].name):
                self.n_fps += 1
                self.fps_scores.append(cfids[0])

        # verify
        if (max_cfid > accept_thres) and (max_idx < 1):
            # accepted!
            verify_result = 1
            # adaptive enrollment
            if (max_cfid > enroll_thres)\
                and (self.sv_mode != 'base'):
                enroll_result = selected_model.enroll(
                        key, in_utter, max_cfid)
                # selected_model.show_enrolls()
                # print("adaptively enrolled with {}".format(max_cfid))
        else:
            # not accepted!
            # negatively enrollment
            if max_cfid < neg_thres:
                self.spk_models.append(
                        spk_model(self.config, spk_name, [key], [in_utter]))
                # print("negatively enrolled({})".format(
                    # spk_name !=self.enrolled_spks[0]))

        return verify_result, enroll_result, max_cfid

    def show_enrolls(self,):
        for model in self.spk_models:
            print(model.name)
            model.show_enrolls()

    def print_benefits(self,):
        print("benefits: {}/{}".format(self.n_benefits, self.n_fps))
        if len(self.fps_scores) > 0:
            print("fps mean score: {}".format(np.mean(self.fps_scores)))

