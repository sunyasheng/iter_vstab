import numpy as np
import tensorflow as tf
from pwc_tab import pwc_opt
from core_warp import dense_image_warp


###
# PWC-Net context network helpers
###
def refine_flow(feat, flow, lvl, name='ctxt'):
    """Post-ptrocess the estimated optical flow using a "context" nn.
    Args:
        feat: Features of the second-to-last layer from the optical flow estimator
        flow: Estimated flow to refine
        lvl: Index of the level
        name: Op scope name
    Ref:
        Per page 4 of paper, section "Context network," traditional flow methods often use contextual information
        to post-process the flow. Thus we employ a sub-network, called the context network, to effectively enlarge
        the receptive field size of each output unit at the desired pyramid level. It takes the estimated flow and
        features of the second last layer from the optical flow estimator and outputs a refined flow.

        The context network is a feed-forward CNN and its design is based on dilated convolutions. It consists of
        7 convolutional layers. The spatial kernel for each convolutional layer is 3×3. These layers have different
        dilation constants. A convolutional layer with a dilation constant k means that an input unit to a filter
        in the layer are k-unit apart from the other input units to the filter in the layer, both in vertical and
        horizontal directions. Convolutional layers with large dilation constants enlarge the receptive field of
        each output unit without incurring a large computational burden. From bottom to top, the dilation constants
        are 1, 2, 4, 8, 16, 1, and 1.

    Ref PyTorch code:
        def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
            return nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                padding=padding, dilation=dilation, bias=True),
            nn.LeakyReLU(0.1))
        def predict_flow(in_planes):
            return nn.Conv2d(in_planes,2,kernel_size=3,stride=1,padding=1,bias=True)
        [...]
        self.dc_conv1 = conv(od+dd[4], 128, kernel_size=3, stride=1, padding=1,  dilation=1)
        self.dc_conv2 = conv(128,      128, kernel_size=3, stride=1, padding=2,  dilation=2)
        self.dc_conv3 = conv(128,      128, kernel_size=3, stride=1, padding=4,  dilation=4)
        self.dc_conv4 = conv(128,      96,  kernel_size=3, stride=1, padding=8,  dilation=8)
        self.dc_conv5 = conv(96,       64,  kernel_size=3, stride=1, padding=16, dilation=16)
        self.dc_conv6 = conv(64,       32,  kernel_size=3, stride=1, padding=1,  dilation=1)
        self.dc_conv7 = predict_flow(32)
        [...]
        x = torch.cat((corr2, c12, up_flow3, up_feat3), 1)
        x = torch.cat((self.conv2_0(x), x),1)
        x = torch.cat((self.conv2_1(x), x),1)
        x = torch.cat((self.conv2_2(x), x),1)
        x = torch.cat((self.conv2_3(x), x),1)
        x = torch.cat((self.conv2_4(x), x),1)
        flow2 = self.predict_flow2(x)
        x = self.dc_conv4(self.dc_conv3(self.dc_conv2(self.dc_conv1(x))))
        flow2 += self.dc_conv7(self.dc_conv6(self.dc_conv5(x)))
    """
    op_name = f'refined_flow{lvl}'
    if pwc_opt.dbg:
        print(f'Adding {op_name} sum of dc_convs_chain({feat.op.name}) with {flow.op.name}')
    init = tf.keras.initializers.he_normal()
    with tf.variable_scope(name):
        x = tf.layers.conv2d(feat, 128, 3, 1, 'same', dilation_rate=1, kernel_initializer=init, name=f'dc_conv{lvl}1')
        x = tf.nn.leaky_relu(x, alpha=0.1)  # default alpha is 0.2 for TF
        x = tf.layers.conv2d(x, 128, 3, 1, 'same', dilation_rate=2, kernel_initializer=init, name=f'dc_conv{lvl}2')
        x = tf.nn.leaky_relu(x, alpha=0.1)
        x = tf.layers.conv2d(x, 128, 3, 1, 'same', dilation_rate=4, kernel_initializer=init, name=f'dc_conv{lvl}3')
        x = tf.nn.leaky_relu(x, alpha=0.1)
        x = tf.layers.conv2d(x, 96, 3, 1, 'same', dilation_rate=8, kernel_initializer=init, name=f'dc_conv{lvl}4')
        x = tf.nn.leaky_relu(x, alpha=0.1)
        x = tf.layers.conv2d(x, 64, 3, 1, 'same', dilation_rate=16, kernel_initializer=init, name=f'dc_conv{lvl}5')
        x = tf.nn.leaky_relu(x, alpha=0.1)
        x = tf.layers.conv2d(x, 32, 3, 1, 'same', dilation_rate=1, kernel_initializer=init, name=f'dc_conv{lvl}6')
        x = tf.nn.leaky_relu(x, alpha=0.1)
        x = tf.layers.conv2d(x, 2, 3, 1, 'same', dilation_rate=1, kernel_initializer=init, name=f'dc_conv{lvl}7')

        return tf.add(flow, x, name=op_name)



###
# PWC-Net warping helpers
###
def warp_net(c2, sc_up_flow, lvl, name='warp'):
    """Warp a level of Image1's feature pyramid using the upsampled flow at level+1 of Image2's pyramid.
    Args:
        c2: The level of the feature pyramid of Image2 to warp
        sc_up_flow: Scaled and upsampled estimated optical flow (from Image1 to Image2) used for warping
        lvl: Index of that level
        name: Op scope name
    Ref:
        Per page 4 of paper, section "Warping layer," at the l-th level, we warp features of the second image toward
        the first image using the x2 upsampled flow from the l+1th level:
            C1w<sup>l</sup>(x) = C2<sup>l</sup>(x + Up2(w<sup>l+1</sup>)(x))
        where x is the pixel index and the upsampled flow Up2(w<sup>l+1</sup>) is set to be zero at the top level.
        We use bilinear interpolation to implement the warping operation and compute the gradients to the input
        CNN features and flow for backpropagation according to E. Ilg's FlowNet 2.0 paper.
        For non-translational motion, warping can compensate for some geometric distortions and put image patches
        at the right scale.

        Per page 3 of paper, section "3. Approach," the warping and cost volume layers have no learnable parameters
        and, hence, reduce the model size.

    Ref PyTorch code:
        # warp an image/tensor (im2) back to im1, according to the optical flow
        # x: [B, C, H, W] (im2)
        # flo: [B, 2, H, W] flow
        def warp(self, x, flo):

            B, C, H, W = x.size()
            # mesh grid
            xx = torch.arange(0, W).view(1,-1).repeat(H,1)
            yy = torch.arange(0, H).view(-1,1).repeat(1,W)
            xx = xx.view(1,1,H,W).repeat(B,1,1,1)
            yy = yy.view(1,1,H,W).repeat(B,1,1,1)
            grid = torch.cat((xx,yy),1).float()

            if x.is_cuda:
                grid = grid.cuda()
            vgrid = Variable(grid) + flo

            # scale grid to [-1,1]
            vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:]/max(W-1,1)-1.0
            vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:]/max(H-1,1)-1.0

            vgrid = vgrid.permute(0,2,3,1)
            output = nn.functional.grid_sample(x, vgrid)
            mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
            mask = nn.functional.grid_sample(mask, vgrid)

            mask[mask<0.9999] = 0
            mask[mask>0] = 1

            return output*mask
        [...]
        warp5 = self.warp(c25, up_flow6*0.625)
        warp4 = self.warp(c24, up_flow5*1.25)
        warp3 = self.warp(c23, up_flow4*2.5)
        warp2 = self.warp(c22, up_flow3*5.0)

    Ref TF documentation:
        tf.contrib.image.dense_image_warp(image, flow, name='dense_image_warp')
        https://www.tensorflow.org/api_docs/python/tf/contrib/image/dense_image_warp
        https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/image/python/kernel_tests/dense_image_warp_test.py

    Other implementations:
        https://github.com/bryanyzhu/deepOF/blob/master/flyingChairsWrapFlow.py
        https://github.com/bryanyzhu/deepOF/blob/master/ucf101wrapFlow.py
        https://github.com/rajat95/Optical-Flow-Warping-Tensorflow/blob/master/warp.py
    """
    op_name = f'{name}{lvl}'
    if pwc_opt.dbg:
        msg = f'Adding {op_name} with inputs {c2.op.name} and {sc_up_flow.op.name}'
        print(msg)
    with tf.name_scope(name):
        return dense_image_warp(c2, sc_up_flow, name=op_name)



def cost_volume(c1, warp, search_range, name):
    """Build cost volume for associating a pixel from Image1 with its corresponding pixels in Image2.
    Args:
        c1: Level of the feature pyramid of Image1
        warp: Warped level of the feature pyramid of image22
        search_range: Search range (maximum displacement)
    """
    padded_lvl = tf.pad(warp, [[0, 0], [search_range, search_range], [search_range, search_range], [0, 0]])
    _, h, w, _ = tf.unstack(tf.shape(c1))
    max_offset = search_range * 2 + 1

    cost_vol = []
    for y in range(0, max_offset):
        for x in range(0, max_offset):
            slice = tf.slice(padded_lvl, [0, y, x, 0], [-1, h, w, -1])
            cost = tf.reduce_mean(c1 * slice, axis=3, keepdims=True)
            cost_vol.append(cost)
    cost_vol = tf.concat(cost_vol, axis=3)
    cost_vol = tf.nn.leaky_relu(cost_vol, alpha=0.1, name=name)

    return cost_vol


def extract_features(x_tnsr, name='featpyr'):
    """Extract pyramid of features
    Args:
        x_tnsr: Input tensor (input pair of images in [batch_size, 2, H, W, 3] format)
        name: Variable scope name
    Returns:
        c1, c2: Feature pyramids
    Ref:
        Per page 3 of paper, section "Feature pyramid extractor," given two input images I1 and I2, we generate
        L-level pyramids of feature representations, with the bottom (zeroth) level being the input images,
        i.e., Ct<sup>0</sup> = It. To generate feature representation at the l-th layer, Ct<sup>l</sup>, we use
        layers of convolutional filters to downsample the features at the (l−1)th pyramid level, Ct<sup>l-1</sup>,
        by a factor of 2. From the first to the sixth levels, the number of feature channels are respectively
        16, 32, 64, 96, 128, and 196. Also see page 15 of paper for a rendering of the network architecture.
        Per page 15, individual images of the image pair are encoded using the same Siamese network. Each
        convolution is followed by a leaky ReLU unit. The convolutional layer and the x2 downsampling layer at
        each level is implemented using a single convolutional layer with a stride of 2.

        Note that Figure 4 on page 15 differs from the PyTorch implementation in two ways:
        - It's missing a convolution layer at the end of each conv block
        - It shows a number of filters of 192 (instead of 196) at the end of the last conv block

    Ref PyTorch code:
        def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
            return nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                padding=padding, dilation=dilation, bias=True), nn.LeakyReLU(0.1))
        [...]
        self.conv1a  = conv(3,   16, kernel_size=3, stride=2)
        self.conv1aa = conv(16,  16, kernel_size=3, stride=1)
        self.conv1b  = conv(16,  16, kernel_size=3, stride=1)

        self.conv2a  = conv(16,  32, kernel_size=3, stride=2)
        self.conv2aa = conv(32,  32, kernel_size=3, stride=1)
        self.conv2b  = conv(32,  32, kernel_size=3, stride=1)

        self.conv3a  = conv(32,  64, kernel_size=3, stride=2)
        self.conv3aa = conv(64,  64, kernel_size=3, stride=1)
        self.conv3b  = conv(64,  64, kernel_size=3, stride=1)

        self.conv4a  = conv(64,  96, kernel_size=3, stride=2)
        self.conv4aa = conv(96,  96, kernel_size=3, stride=1)
        self.conv4b  = conv(96,  96, kernel_size=3, stride=1)

        self.conv5a  = conv(96, 128, kernel_size=3, stride=2)
        self.conv5aa = conv(128,128, kernel_size=3, stride=1)
        self.conv5b  = conv(128,128, kernel_size=3, stride=1)

        self.conv6aa = conv(128,196, kernel_size=3, stride=2)
        self.conv6a  = conv(196,196, kernel_size=3, stride=1)
        self.conv6b  = conv(196,196, kernel_size=3, stride=1)
        [...]
        c11 = self.conv1b(self.conv1aa(self.conv1a(im1))) # Higher-res
        c21 = self.conv1b(self.conv1aa(self.conv1a(im2)))
        c12 = self.conv2b(self.conv2aa(self.conv2a(c11)))
        c22 = self.conv2b(self.conv2aa(self.conv2a(c21)))
        c13 = self.conv3b(self.conv3aa(self.conv3a(c12)))
        c23 = self.conv3b(self.conv3aa(self.conv3a(c22)))
        c14 = self.conv4b(self.conv4aa(self.conv4a(c13)))
        c24 = self.conv4b(self.conv4aa(self.conv4a(c23)))
        c15 = self.conv5b(self.conv5aa(self.conv5a(c14)))
        c25 = self.conv5b(self.conv5aa(self.conv5a(c24)))
        c16 = self.conv6b(self.conv6a(self.conv6aa(c15)))
        c26 = self.conv6b(self.conv6a(self.conv6aa(c25))) # Lower-res

    Ref Caffee code:
        https://github.com/NVlabs/PWC-Net/blob/438ca897ae77e08f419ddce5f0d7fa63b0a27a77/Caffe/model/train.prototxt#L314-L1141
    """
    assert (1 <= pwc_opt.pyr_lvls <= 6)
    if pwc_opt.dbg:
        print(f"Building feature pyramids (c11,c21) ... (c1{pwc_opt.pyr_lvls},c2{pwc_opt.pyr_lvls})")
    # Make the feature pyramids 1-based for better readability down the line
    # [x_tnsr1, x_tnsr2] = x_tnsr
    num_chann = [None, 16, 32, 64, 96, 128, 196]
    c1, c2 = [None], [None]
    init = tf.keras.initializers.he_normal()
    with tf.variable_scope(name):
        for pyr, x, reuse, name in zip([c1, c2], [x_tnsr[:, 0], x_tnsr[:, 1]], [None, True], ['c1', 'c2']):
            for lvl in range(1, pwc_opt.pyr_lvls + 1):
                # tf.layers.conv2d(inputs, filters, kernel_size, strides=(1, 1), padding='valid', ... , name, reuse)
                # reuse is set to True because we want to learn a single set of weights for the pyramid
                # kernel_initializer = 'he_normal' or tf.keras.initializers.he_normal(seed=None)
                f = num_chann[lvl]
                x = tf.layers.conv2d(x, f, 3, 2, 'same', kernel_initializer=init, name=f'conv{lvl}a', reuse=reuse)
                x = tf.nn.leaky_relu(x, alpha=0.1)  # , name=f'relu{lvl+1}a') # default alpha is 0.2 for TF
                x = tf.layers.conv2d(x, f, 3, 1, 'same', kernel_initializer=init, name=f'conv{lvl}aa', reuse=reuse)
                x = tf.nn.leaky_relu(x, alpha=0.1)  # , name=f'relu{lvl+1}aa')
                x = tf.layers.conv2d(x, f, 3, 1, 'same', kernel_initializer=init, name=f'conv{lvl}b', reuse=reuse)
                x = tf.nn.leaky_relu(x, alpha=0.1, name=f'{name}{lvl}')
                pyr.append(x)
    return c1, c2

###
# Cost Volume helpers
###
def corr_net(c1, warp, lvl, name='corr'):
    """Build cost volume for associating a pixel from Image1 with its corresponding pixels in Image2.
    Args:
        c1: The level of the feature pyramid of Image1
        warp: The warped level of the feature pyramid of image22
        lvl: Index of that level
        name: Op scope name
    Ref:
        Per page 3 of paper, section "Cost Volume," a cost volume stores the data matching costs for associating
        a pixel from Image1 with its corresponding pixels in Image2. Most traditional optical flow techniques build
        the full cost volume at a single scale, which is both computationally expensive and memory intensive. By
        contrast, PWC-Net constructs a partial cost volume at multiple pyramid levels.

        The matching cost is implemented as the correlation between features of the first image and warped features
        of the second image:
            CV<sup>l</sup>(x1,x2) = (C1<sup>l</sup>(x1))<sup>T</sup> . Cw<sup>l</sup>(x2) / N
        where where T is the transpose operator and N is the length of the column vector C1<sup>l</sup>(x1).
        For an L-level pyramid, we only need to compute a partial cost volume with a limited search range of d
        pixels. A one-pixel motion at the top level corresponds to 2**(L−1) pixels at the full resolution images.
        Thus we can set d to be small, e.g. d=4. The dimension of the 3D cost volume is d**2 × Hl × Wl, where Hl
        and Wl denote the height and width of the L-th pyramid level, respectively.

        Per page 3 of paper, section "3. Approach," the warping and cost volume layers have no learnable parameters
        and, hence, reduce the model size.

        Per page 5 of paper, section "Implementation details," we use a search range of 4 pixels to compute the
        cost volume at each level.

    Ref PyTorch code:
    from correlation_package.modules.corr import Correlation
    self.corr = Correlation(pad_size=md, kernel_size=1, max_displacement=4, stride1=1, stride2=1, corr_multiply=1)
    [...]
    corr6 = self.corr(c16, c26)
    corr6 = self.leakyRELU(corr6)
    ...
    corr5 = self.corr(c15, warp5)
    corr5 = self.leakyRELU(corr5)
    ...
    corr4 = self.corr(c14, warp4)
    corr4 = self.leakyRELU(corr4)
    ...
    corr3 = self.corr(c13, warp3)
    corr3 = self.leakyRELU(corr3)
    ...
    corr2 = self.corr(c12, warp2)
    corr2 = self.leakyRELU(corr2)
    """
    op_name = f'corr{lvl}'
    if pwc_opt.dbg:
        print(f'Adding {op_name} with inputs {c1.op.name} and {warp.op.name}')
    with tf.name_scope(name):
        return cost_volume(c1, warp, pwc_opt.search_range, op_name)

def predict_flow(corr, c1, up_flow, up_feat, lvl, name='predict_flow'):
    """Estimate optical flow.
    Args:
        corr: The cost volume at level lvl
        c1: The level of the feature pyramid of Image1
        up_flow: An upsampled version of the predicted flow from the previous level
        up_feat: An upsampled version of the features that were used to generate the flow prediction
        lvl: Index of the level
        name: Op scope name
    Args:
        upfeat: The features used to generate the predicted flow
        flow: The predicted flow
    Ref:
        Per page 4 of paper, section "Optical flow estimator," the optical flow estimator is a multi-layer CNN. Its
        input are the cost volume, features of the first image, and upsampled optical flow and its output is the
        flow w<sup>l</sup> at the l-th level. The numbers of feature channels at each convolutional layers are
        respectively 128, 128, 96, 64, and 32, which are kept fixed at all pyramid levels. The estimators at
        different levels have their own parameters instead of sharing the same parameters. This estimation process
        is repeated until the desired level, l0.

        Per page 5 of paper, section "Implementation details," we use a 7-level pyramid and set l0 to be 2, i.e.,
        our model outputs a quarter resolution optical flow and uses bilinear interpolation to obtain the
        full-resolution optical flow.

        The estimator architecture can be enhanced with DenseNet connections. The inputs to every convolutional
        layer are the output of and the input to its previous layer. DenseNet has more direct connections than
        traditional layers and leads to significant improvement in image classification.

        Note that we do not use DenseNet connections in this implementation because a) they increase the size of the
        model, and, b) per page 7 of paper, section "Optical flow estimator," removing the DenseNet connections
        results in higher training error but lower validation errors when the model is trained on FlyingChairs
        (that being said, after the model is fine-tuned on FlyingThings3D, DenseNet leads to lower errors).

    Ref PyTorch code:
        def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
            return nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                padding=padding, dilation=dilation, bias=True),
            nn.LeakyReLU(0.1))
        def predict_flow(in_planes):
            return nn.Conv2d(in_planes,2,kernel_size=3,stride=1,padding=1,bias=True)
        [...]
        nd = (2*md+1)**2
        dd = np.cumsum([128,128,96,64,32])
        od = nd
        self.conv6_0 = conv(od,      128, kernel_size=3, stride=1)
        self.conv6_1 = conv(od+dd[0],128, kernel_size=3, stride=1)
        self.conv6_2 = conv(od+dd[1],96,  kernel_size=3, stride=1)
        self.conv6_3 = conv(od+dd[2],64,  kernel_size=3, stride=1)
        self.conv6_4 = conv(od+dd[3],32,  kernel_size=3, stride=1)
        self.predict_flow6 = predict_flow(od+dd[4])
        [...]
        od = nd+128+4
        self.conv5_0 = conv(od,      128, kernel_size=3, stride=1)
        self.conv5_1 = conv(od+dd[0],128, kernel_size=3, stride=1)
        self.conv5_2 = conv(od+dd[1],96,  kernel_size=3, stride=1)
        self.conv5_3 = conv(od+dd[2],64,  kernel_size=3, stride=1)
        self.conv5_4 = conv(od+dd[3],32,  kernel_size=3, stride=1)
        self.predict_flow5 = predict_flow(od+dd[4])
        [...]
        od = nd+96+4
        self.conv4_0 = conv(od,      128, kernel_size=3, stride=1)
        self.conv4_1 = conv(od+dd[0],128, kernel_size=3, stride=1)
        self.conv4_2 = conv(od+dd[1],96,  kernel_size=3, stride=1)
        self.conv4_3 = conv(od+dd[2],64,  kernel_size=3, stride=1)
        self.conv4_4 = conv(od+dd[3],32,  kernel_size=3, stride=1)
        self.predict_flow4 = predict_flow(od+dd[4])
        [...]
        od = nd+64+4
        self.conv3_0 = conv(od,      128, kernel_size=3, stride=1)
        self.conv3_1 = conv(od+dd[0],128, kernel_size=3, stride=1)
        self.conv3_2 = conv(od+dd[1],96,  kernel_size=3, stride=1)
        self.conv3_3 = conv(od+dd[2],64,  kernel_size=3, stride=1)
        self.conv3_4 = conv(od+dd[3],32,  kernel_size=3, stride=1)
        self.predict_flow3 = predict_flow(od+dd[4])
        [...]
        od = nd+32+4
        self.conv2_0 = conv(od,      128, kernel_size=3, stride=1)
        self.conv2_1 = conv(od+dd[0],128, kernel_size=3, stride=1)
        self.conv2_2 = conv(od+dd[1],96,  kernel_size=3, stride=1)
        self.conv2_3 = conv(od+dd[2],64,  kernel_size=3, stride=1)
        self.conv2_4 = conv(od+dd[3],32,  kernel_size=3, stride=1)
        self.predict_flow2 = predict_flow(od+dd[4])
        [...]
        self.dc_conv1 = conv(od+dd[4], 128, kernel_size=3, stride=1, padding=1,  dilation=1)
        self.dc_conv2 = conv(128,      128, kernel_size=3, stride=1, padding=2,  dilation=2)
        self.dc_conv3 = conv(128,      128, kernel_size=3, stride=1, padding=4,  dilation=4)
        self.dc_conv4 = conv(128,      96,  kernel_size=3, stride=1, padding=8,  dilation=8)
        self.dc_conv5 = conv(96,       64,  kernel_size=3, stride=1, padding=16, dilation=16)
        self.dc_conv6 = conv(64,       32,  kernel_size=3, stride=1, padding=1,  dilation=1)
        self.dc_conv7 = predict_flow(32)
        [...]
        x = torch.cat((self.conv6_0(corr6), corr6),1)
        x = torch.cat((self.conv6_1(x), x),1)
        x = torch.cat((self.conv6_2(x), x),1)
        x = torch.cat((self.conv6_3(x), x),1)
        x = torch.cat((self.conv6_4(x), x),1)
        flow6 = self.predict_flow6(x)
        ...
        x = torch.cat((corr5, c15, up_flow6, up_feat6), 1)
        x = torch.cat((self.conv5_0(x), x),1)
        x = torch.cat((self.conv5_1(x), x),1)
        x = torch.cat((self.conv5_2(x), x),1)
        x = torch.cat((self.conv5_3(x), x),1)
        x = torch.cat((self.conv5_4(x), x),1)
        flow5 = self.predict_flow5(x)
        ...
        x = torch.cat((corr4, c14, up_flow5, up_feat5), 1)
        x = torch.cat((self.conv4_0(x), x),1)
        x = torch.cat((self.conv4_1(x), x),1)
        x = torch.cat((self.conv4_2(x), x),1)
        x = torch.cat((self.conv4_3(x), x),1)
        x = torch.cat((self.conv4_4(x), x),1)
        flow4 = self.predict_flow4(x)
        ...
        x = torch.cat((corr3, c13, up_flow4, up_feat4), 1)
        x = torch.cat((self.conv3_0(x), x),1)
        x = torch.cat((self.conv3_1(x), x),1)
        x = torch.cat((self.conv3_2(x), x),1)
        x = torch.cat((self.conv3_3(x), x),1)
        x = torch.cat((self.conv3_4(x), x),1)
        flow3 = self.predict_flow3(x)
        ...
        x = torch.cat((corr2, c12, up_flow3, up_feat3), 1)
        x = torch.cat((self.conv2_0(x), x),1)
        x = torch.cat((self.conv2_1(x), x),1)
        x = torch.cat((self.conv2_2(x), x),1)
        x = torch.cat((self.conv2_3(x), x),1)
        x = torch.cat((self.conv2_4(x), x),1)
        flow2 = self.predict_flow2(x)
    """
    op_name = f'flow{lvl}'
    init = tf.keras.initializers.he_normal()
    with tf.variable_scope(name):
        if c1 is None and up_flow is None and up_feat is None:
            if pwc_opt.dbg:
                print(f'Adding {op_name} with input {corr.op.name}')
            x = corr
        else:
            if pwc_opt.dbg:
                msg = f'Adding {op_name} with inputs {corr.op.name}, {c1.op.name}, {up_flow.op.name}, {up_feat.op.name}'
                print(msg)
            x = tf.concat([corr, c1, up_flow, up_feat], axis=3)

        conv = tf.layers.conv2d(x, 128, 3, 1, 'same', kernel_initializer=init, name=f'conv{lvl}_0')
        act = tf.nn.leaky_relu(conv, alpha=0.1)  # default alpha is 0.2 for TF
        x = tf.concat([act, x], axis=3) if pwc_opt.use_dense_cx else act

        conv = tf.layers.conv2d(x, 128, 3, 1, 'same', kernel_initializer=init, name=f'conv{lvl}_1')
        act = tf.nn.leaky_relu(conv, alpha=0.1)
        x = tf.concat([act, x], axis=3) if pwc_opt.use_dense_cx else act

        conv = tf.layers.conv2d(x, 96, 3, 1, 'same', kernel_initializer=init, name=f'conv{lvl}_2')
        act = tf.nn.leaky_relu(conv, alpha=0.1)
        x = tf.concat([act, x], axis=3) if pwc_opt.use_dense_cx else act

        conv = tf.layers.conv2d(x, 64, 3, 1, 'same', kernel_initializer=init, name=f'conv{lvl}_3')
        act = tf.nn.leaky_relu(conv, alpha=0.1)
        x = tf.concat([act, x], axis=3) if pwc_opt.use_dense_cx else act

        conv = tf.layers.conv2d(x, 32, 3, 1, 'same', kernel_initializer=init, name=f'conv{lvl}_4')
        act = tf.nn.leaky_relu(conv, alpha=0.1)  # will also be used as an input by the context network
        upfeat = tf.concat([act, x], axis=3, name=f'upfeat{lvl}') if pwc_opt.use_dense_cx else act

        flow = tf.layers.conv2d(upfeat, 2, 3, 1, 'same', name=op_name)

        return upfeat, flow

def deconv(x, lvl, name='up_flow'):
    """Upsample, not using a bilinear filter, but rather learn the weights of a conv2d_transpose op filters.
    Args:
        x: Level features or flow to upsample
        lvl: Index of that level
        name: Op scope name
    Ref PyTorch code:
        def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
            return nn.ConvTranspose2d(in_planes, out_planes, kernel_size, stride, padding, bias=True)
        [...]
        self.deconv6 = deconv(2, 2, kernel_size=4, stride=2, padding=1)
        self.upfeat6 = deconv(od+dd[4], 2, kernel_size=4, stride=2, padding=1)
        ...
        self.deconv5 = deconv(2, 2, kernel_size=4, stride=2, padding=1)
        self.upfeat5 = deconv(od+dd[4], 2, kernel_size=4, stride=2, padding=1)
        ...
        self.deconv4 = deconv(2, 2, kernel_size=4, stride=2, padding=1)
        self.upfeat4 = deconv(od+dd[4], 2, kernel_size=4, stride=2, padding=1)
        ...
        self.deconv3 = deconv(2, 2, kernel_size=4, stride=2, padding=1)
        self.upfeat3 = deconv(od+dd[4], 2, kernel_size=4, stride=2, padding=1)
        ...
        self.deconv2 = deconv(2, 2, kernel_size=4, stride=2, padding=1)
        [...]
        up_flow6 = self.deconv6(flow6)
        up_feat6 = self.upfeat6(x)
        ...
        up_flow5 = self.deconv5(flow5)
        up_feat5 = self.upfeat5(x)
        ...
        up_flow4 = self.deconv4(flow4)
        up_feat4 = self.upfeat4(x)
        ...
        up_flow3 = self.deconv3(flow3)
        up_feat3 = self.upfeat3(x)
    """
    op_name = f'{name}{lvl}'
    if pwc_opt.dbg:
        print(f'Adding {op_name} with input {x.op.name}')
    with tf.variable_scope('upsample'):
        # tf.layers.conv2d_transpose(inputs, filters, kernel_size, strides=(1, 1), padding='valid', ... , name)
        return tf.layers.conv2d_transpose(x, 2, 4, 2, 'same', name=op_name)

def nn(x_tnsr, name='pwcnet', reuse=False):
    """Defines and connects the backbone neural nets
    Args:
        inputs: TF placeholder that contains the input frame pairs in [batch_size, 2, H, W, 3] format
        name: Name of the nn
    Returns:
        net: Output tensors of the backbone network
    Ref:
        RE: the scaling of the upsampled estimated optical flow, per page 5, section "Implementation details," we
        do not further scale the supervision signal at each level, the same as the FlowNet paper. As a result, we
        need to scale the upsampled flow at each pyramid level for the warping layer. For example, at the second
        level, we scale the upsampled flow from the third level by a factor of 5 (=20/4) before warping features
        of the second image.
    Based on:
        - https://github.com/daigo0927/PWC-Net_tf/blob/master/model.py
        Written by Daigo Hirooka, Copyright (c) 2018 Daigo Hirooka
        MIT License
    """
    with tf.variable_scope(name, reuse=reuse):

        # Extract pyramids of CNN features from both input images (1-based lists))
        c1, c2 = extract_features(x_tnsr)

        flow_pyr = []

        for lvl in range(pwc_opt.pyr_lvls, pwc_opt.flow_pred_lvl - 1, -1):

            if lvl == pwc_opt.pyr_lvls:
                # Compute the cost volume
                corr = corr_net(c1[lvl], c2[lvl], lvl)

                # Estimate the optical flow
                upfeat, flow = predict_flow(corr, None, None, None, lvl)
            else:
                # Warp level of Image1's using the upsampled flow
                scaler = 20. / 2**lvl  # scaler values are 0.625, 1.25, 2.5, 5.0
                warp = warp_net(c2[lvl], up_flow * scaler, lvl)

                # Compute the cost volume
                corr = corr_net(c1[lvl], warp, lvl)

                # Estimate the optical flow
                upfeat, flow = predict_flow(corr, c1[lvl], up_flow, up_feat, lvl)

            _, lvl_height, lvl_width, _ = tf.unstack(tf.shape(c1[lvl]))

            if lvl != pwc_opt.flow_pred_lvl:
                if pwc_opt.use_res_cx:
                    flow = refine_flow(upfeat, flow, lvl)

                # Upsample predicted flow and the features used to compute predicted flow
                flow_pyr.append(flow)

                up_flow = deconv(flow, lvl, 'up_flow')
                up_feat = deconv(upfeat, lvl, 'up_feat')
            else:
                # Refine the final predicted flow
                flow = refine_flow(upfeat, flow, lvl)
                flow_pyr.append(flow)

                # Upsample the predicted flow (final output) to match the size of the images
                scaler = 2**pwc_opt.flow_pred_lvl
                size = (lvl_height * scaler, lvl_width * scaler)
                flow_pred = tf.image.resize_bilinear(flow, size, name="flow_pred") * scaler
                break

        return flow_pred, flow_pyr

if __name__ == '__main__':
    x_tnsr = tf.placeholder(dtype=tf.float32, shape=[None, 2, None, None, 3])
    flow_pred, flow_pyr = nn(x_tnsr)
    flow_pred = flow_pred[:, :, :, ::-1]
    warped_img1 = tf.contrib.image.dense_image_warp(image=x_tnsr[:, 0], flow = flow_pred*0.5)
    saver = tf.train.Saver(max_to_keep=200)
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))

    import cv2
    from train import optimistic_restore
    def pad_img(img, pyr_lvls):
        ##img shape: [h, w, c]
        _, pad_h = divmod(img.shape[0], 2**pyr_lvls)
        if pad_h != 0:
            pad_h = 2 ** pyr_lvls - pad_h
        _, pad_w = divmod(img.shape[1], 2**pyr_lvls)
        if pad_w != 0:
            pad_w = 2 ** pyr_lvls - pad_w
        if pad_h != 0 or pad_w != 0:
            padding = [(0, pad_h), (0, pad_w), (0, 0)]
            img = np.pad(img, padding, mode='constant', constant_values=0.)
        return img

    # import os
    # print(os.path.exists(pwc_opt.ckpt_path))
    optimistic_restore(sess, pwc_opt.ckpt_path)
    # saver.restore(sess, pwc_opt.ckpt_path)
    img_path1, img_path2 = f'middlebury/Beanbags/frame10.png', f'middlebury/Beanbags/frame11.png'
    image1, image2 = cv2.imread(img_path1), cv2.imread(img_path2)
    image1, image2 = pad_img(image1, 6), pad_img(image2, 6)
    print(image1.shape, image2.shape)
    resized_img1, resized_img2 = image1, image2
    # resized_img1, resized_img2 = cv2.resize(image1, (256, 256)), cv2.resize(image2, (256, 256))
    resized_img1, resized_img2 = np.expand_dims(resized_img1, axis=0), np.expand_dims(resized_img2, axis=0)
    # import pdb; pdb.set_trace();
    warped_img1_np = sess.run(warped_img1, feed_dict={x_tnsr: np.stack([resized_img1/255., resized_img2/255.], axis=1)})
    print(warped_img1_np.shape)
    print(warped_img1_np.mean())
    cv2.imshow('warped ', warped_img1_np[0])
    cv2.waitKey(-1)
    pass