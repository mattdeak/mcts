from mcts.nn.utils import crossentropy_with_logits
from keras.utils.generic_utils import get_custom_objects

get_custom_objects().update({"crossentropy_with_logits": crossentropy_with_logits})
