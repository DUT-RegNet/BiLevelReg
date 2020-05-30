from modules import *
import neuron.layers as nrn_layers

class nn_trf(object):

    def __init__(self, name='nn_trf'):
        self.name = name

    def __call__(self, images_1_seg, flow, indexing='ij'):
        out = nrn_layers.SpatialTransformer(interp_method='nearest', indexing=indexing)([images_1_seg, flow])
        return out

class BiLevelNet(object):
    GRAD_IS_ZERO = 1e-12

    def __init__(self, num_levels=2, name='pro-net'):
        self.num_levels = num_levels
        self.output_level = num_levels - 1
        self.name = name
        self.fp_extractor = Pro_FeatureLearning(self.num_levels)
        self.of_estimator = [Estimator_1(name=f'estimator1_{l}') for l in range(self.num_levels)]
        self.conv_block = conv_block(name='conv_block')
        self.int_steps = 7

    def __call__(self, images_0, images_1, reuse=False):# reuse=False
        with tf.variable_scope(self.name, reuse=reuse) as vs:
            pyramid_0, pyramid_params_0 = self.fp_extractor(images_0, reuse=reuse)
            pyramid_1, pyramid_params_1 = self.fp_extractor(images_1)
            flows_pyramid = []
            flows_up, features_up = None, None

            for l, (features_0, features_1) in enumerate(zip(pyramid_0, pyramid_1)):

                # Flow estimation
                flows = self.of_estimator[l](features_0, features_1, flows_up)

                # Integrate if diffeomorphic (i.e. treating 'flow' above as stationary velocity field)
                z_sample = flows
                flows = nrn_layers.VecInt(method='ss', name='flow-int', int_steps=self.int_steps)(z_sample)

                if l < self.output_level:
                    # up-sample
                    flows_up = nrn_layers.Resize(zoom_factor=2, interp_method='linear')(flows*2)
                else:
                    # At output level
                    flows_pyramid.append(flows)
                    # Obtain finally scale-adjusted flow
                    upscale = 2**(self.num_levels-self.output_level)
                    flows_final = nrn_layers.Resize(zoom_factor=upscale, interp_method='linear')(flows*upscale)
                    y = nrn_layers.SpatialTransformer(interp_method='linear', indexing='ij')([images_1, flows_final])
                    return flows_final, y

                flows_pyramid.append(flows)

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]
