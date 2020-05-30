import argparse
from utils import *
from model import *

class Tester(object):
    def __init__(self, args):
        self.args = args
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config = config)
        self._build_graph()

    def _build_graph(self):
        self.images_tf = tf.placeholder(tf.float32, shape=(1, 2, 160, 192, 224, 1))
        self.model = BiLevelNet(num_levels=2)
        self.flows_final, self.y = self.model(self.images_tf[:, 0], self.images_tf[:, 1])  # image0 image1 (fixed and moving)
        self.nn_trf = nn_trf(name='nn_trf')
        self.saver = tf.train.Saver()
        if self.args.resume is not None:
            print(f'Loading learned model from checkpoint {self.args.resume}')
            self.saver.restore(self.sess, self.args.resume)
        else:
            print('!!! Test with un-learned model !!!')
            self.sess.run(tf.global_variables_initializer())

    def test(self):
        img1 = load_nii('data\\atlas_norm.nii.gz')
        img2= load_nii(self.args.vol_name)
        images = np.array([img1, img2])  # shape(2, h, w, c, 1)
        images = np.reshape(images, (1,) + images.shape)  # shape(1, 2, h, w, c, 1)

        # Save Vol
        vol = self.sess.run(self.y, feed_dict={self.images_tf: images})
        vol = nib.Nifti1Image(vol[0, :, :, :, :], np.eye(4))
        nib.save(vol, 'data\warped_vol.nii.gz')



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=str, default='Model\model_40.ckpt',
                        help='Learned parameter checkpoint file [None]')
    parser.add_argument('--vol_name', type=str, default="data\\brain_scan1_vol.nii.gz")
    args = parser.parse_args()
    for key, item in vars(args).items():
        print(f'{key} : {item}')
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    tester = Tester(args)
    tester.test()



