import time
import math
import torch
from torch import nn
from tqdm import tqdm
import torch.nn.functional as F

EMA_CPU = None


class EMA(nn.Module):
    def __init__(self, model, decay=0.9999, use_num_upates=True):
        super().__init__()
        if decay < 0.0 or decay > 1.0:
            raise ValueError('Decay must be between 0 and 1')

        self.m_name2s_name = {}
        self.collected_params = {}
        self.register_buffer('decay', torch.tensor(decay, dtype=torch.float32))
        self.register_buffer('num_updates', torch.tensor(0, dtype=torch.int) if use_num_upates else torch.tensor(-1, dtype=torch.int))
        for name, p in model.named_parameters():
            if p.requires_grad:
                #remove as '.'-character is not allowed in buffers
                s_name = name.replace('.', '')
                self.m_name2s_name.update({name: s_name})
                self.register_buffer(s_name, p.clone().detach().data)
                self.collected_params["_coll_params_"+s_name] = p.detach().data.clone()

    def forward(self, model):
        with torch.no_grad():
            decay = self.decay
            if self.num_updates >= 0:
                self.num_updates += 1
                decay = min(self.decay,
                            (1 + self.num_updates) / (10 + self.num_updates))
            one_minus_decay = 1.0 - decay
            # with torch.no_grad():
            m_param = dict(model.named_parameters())
            shadow_params = dict(self.named_buffers())

            for key in m_param:
                if m_param[key].requires_grad:
                    sname = self.m_name2s_name[key]
                    shadow_params[sname] = shadow_params[sname].type_as(
                        m_param[key])
                    shadow_params[sname].sub_(
                        one_minus_decay *
                        (shadow_params[sname] - m_param[key].data))
                else:
                    assert key not in self.m_name2s_name

    def copy_to(self, model):
        """
        Copy smooth weight the online model
        """
        # Fast updating weights
        m_param = dict(model.named_parameters())
        # Emponential Averaged Weights
        shadow_params = dict(self.named_buffers())
        for key in m_param:
            if m_param[key].requires_grad:
                m_param[key].data.copy_(shadow_params[self.m_name2s_name[key]].data)
            else:
                assert key not in self.m_name2s_name

    def store(self, model):
        """
        Save the current parameters for restoring later.
        """
        # self.collected_params = {}
        m_param = dict(model.named_parameters())
        # self.collected_params = dict(self.named_buffers())
        for key in m_param:
            if m_param[key].requires_grad:
                sname = self.m_name2s_name[key]
                self.collected_params["_coll_params_"+sname] = m_param[key].data.clone()
            else:
                assert key not in self.m_name2s_name

    def restore(self, model):
        m_param = dict(model.named_parameters())
        for key in m_param:
            if m_param[key].requires_grad:
                sname = self.m_name2s_name[key]
                m_param[key].data.copy_(self.collected_params["_coll_params_" +sname])
            else:
                assert key not in self.m_name2s_name
        # self.collected_params = {}

