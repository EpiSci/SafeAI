import tensorflow as tf
from safeai.models import joint_confident
mnist = tf.keras.datasets.mnist

def dcgan():
    pass

def vgg():
    pass

def main():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    image_width, image_height = x_train.shape[1:]
    image_feature = tf.feature_column.numeric_column('image', shape=[image_width, image_height])
    
    joint_confident_classifier = tf.estimator.Estimator(
        model_fn=joint_confident,
        params={
            'feature_columns': [image_feature],
            'num_classes': 10,
            'discriminator_fn': None,
            'generator_fn': None,
            'classifier_fn': None,
        }) 

if __name__ == "__main__":
    main()
