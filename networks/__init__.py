from iris.networks.vgg import VGG
from iris.networks.mlp import MLP

catalogue = dict()


def register(cls):
    catalogue.update({cls.name: cls})


register(VGG)
register(MLP)
