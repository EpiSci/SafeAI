
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from safeai.models import joint_confident

import tensorflow as tf

class JointConfidentModelTest(tf.test.TestCase):
    
    def test_confident_classifier(self):
        with self.cached_session():
            x = tf.square([2, 3])
            self.assertAllEqual(x.eval(), [4, 9])

if __name__ == "__main__":
    tf.test.main()
