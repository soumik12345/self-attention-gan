from sagan import SelfAttentionGAN
from configs.base import get_config


gan = SelfAttentionGAN(configs=get_config())
gan.summary()
