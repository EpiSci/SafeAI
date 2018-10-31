import tensorflow as tf
import pytest

import safeai.models.joint_confident as jc

@pytest.mark.filterwarnings("ignore:DeprecationWarning")
def test_sample_model():
    one = jc.sample_model()
    assert one == 1
