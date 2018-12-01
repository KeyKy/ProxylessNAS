import os
import json

import torch

from proxyless_nas.utils import download_url, load_url
from .nas_modules import MobilenetV1


def load_init(expdir):
    init_path = '%s/init' % expdir
    assert os.path.isfile(init_path)

    if torch.cuda.is_available():
        checkpoint = torch.load(init_path)
    else:
        checkpoint = torch.load(init_path, map_location='cpu')
    return checkpoint


def load_net(expdir, print_info=False):
    net_config_path = '%s/net.config' % expdir
    assert os.path.isfile(net_config_path), 'No net configs found in <%s>' % expdir
    net_config_json = json.load(open(net_config_path, 'r'))
    if print_info:
        print('Net config:')
        for k, v in net_config_json.items():
            if k != 'blocks':
                print('\t%s: %s' % (k, v))
    net = MobilenetV1.build_from_config(net_config_json)
    if 'bn' in net_config_json:
        net.set_bn_param(bn_momentum=net_config_json['bn']['momentum'], bn_eps=net_config_json['bn']['eps'])
    else:
        net.set_bn_param(bn_momentum=0.1, bn_eps=1e-3)
    return net


def proxyless_base(pretrained=True, net_config=None, net_weight=None):

    assert net_config is not None, "Please input a network config"
    net_config_path = download_url(net_config)
    net_config_json = json.load(open(net_config_path, 'r'))
    net = MobilenetV1.build_from_config(net_config_json)
    if 'bn' in net_config_json:
        net.set_bn_param(bn_momentum=net_config_json['bn']['momentum'], bn_eps=net_config_json['bn']['eps'])
    else:
        net.set_bn_param(bn_momentum=0.1, bn_eps=1e-3)

    if pretrained:
        assert net_weight is not None, "Please specify network weights"
        init_path = download_url(net_weight)
        init = torch.load(init_path, map_location='cpu')
        net.load_state_dict(init['state_dict'])

    return net

from functools import partial

proxyless_cpu = partial(proxyless_base,
                        net_config="http://hanlab.mit.edu/files/proxylessNAS/proxyless_cpu.config",
                        net_weight="http://hanlab.mit.edu/files/proxylessNAS/proxyless_cpu.pth")

proxyless_gpu = partial(proxyless_base,
                        net_config="http://hanlab.mit.edu/files/proxylessNAS/proxyless_gpu.config",
                        net_weight="http://hanlab.mit.edu/files/proxylessNAS/proxyless_gpu.pth")

proxyless_mobile = partial(proxyless_base,
                        net_config="http://hanlab.mit.edu/files/proxylessNAS/proxyless_mobile.config",
                        net_weight="http://hanlab.mit.edu/files/proxylessNAS/proxyless_mobile.pth")