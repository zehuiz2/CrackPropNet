import torch
import torch.nn as nn
from .FlowNet2CSS import FlowNet2CSS
from .util import conv


class CrackPropNet(FlowNet2CSS):
    def __init__(self, batchNorm=False, div_flow=20.):
        super(CrackPropNet,self).__init__(batchNorm=batchNorm, div_flow=20.)
        self.batchNorm = batchNorm
        self.div_flow = div_flow
        self.rgb_max = 1.0

        self.conv7_1 = conv(True, 194, 256)
        self.conv7_2 = conv(True, 256, 256)
        self.conv7_3 = conv(True, 256, 256)

        self.conv8_1 = conv(True, 256, 512, stride=2)
        self.conv8_2 = conv(True, 512, 512)
        self.conv8_3 = conv(True, 512, 512)

        self.conv7_down = conv(True, 256, 21, kernel_size=1, stride=2)
        self.conv8_down = conv(True, 512, 21, kernel_size=1, stride=1)

        self.conv_final = nn.Conv2d(42, 1, kernel_size=1, stride=1, bias=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    nn.init.uniform_(m.bias)
                nn.init.kaiming_normal_(m.weight)

            if isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    nn.init.uniform_(m.bias)
                nn.init.kaiming_normal_(m.weight)

    def forward(self, inputs):
        rgb_mean = inputs.contiguous().view(inputs.size()[:2]+(-1,)).mean(dim=-1).view(inputs.size()[:2] + (1,1,1,))
        
        x = (inputs - rgb_mean) / self.rgb_max
        x1 = x[:,:,0,:,:]
        x2 = x[:,:,1,:,:]
        x = torch.cat((x1,x2), dim = 1)

        # flownetc
        flownetc_flow2 = self.flownetc(x)[0]
        flownetc_flow = self.upsample1(flownetc_flow2*self.div_flow)
        
        # warp img1 to img0; magnitude of diff between img0 and and warped_img1, 
        resampled_img1 = self.resample1(x[:,3:,:,:], flownetc_flow)
        diff_img0 = x[:,:3,:,:] - resampled_img1 
        norm_diff_img0 = self.channelnorm(diff_img0)

        # concat img0, img1, img1->img0, flow, diff-mag ; 
        concat1 = torch.cat((x, resampled_img1, flownetc_flow/self.div_flow, norm_diff_img0), dim=1)
        
        # flownets1
        flownets1_flow2 = self.flownets_1(concat1)[0]
        flownets1_flow = self.upsample2(flownets1_flow2*self.div_flow) 

        # warp img1 to img0 using flownets1; magnitude of diff between img0 and and warped_img1
        resampled_img1 = self.resample2(x[:,3:,:,:], flownets1_flow)
        diff_img0 = x[:,:3,:,:] - resampled_img1
        norm_diff_img0 = self.channelnorm(diff_img0)

        # concat img0, img1, img1->img0, flow, diff-mag
        concat2 = torch.cat((x, resampled_img1, flownets1_flow/self.div_flow, norm_diff_img0), dim=1)

        # flownets2
        flownets2_concat2 = self.flownets_2(concat2)[-1]
        # flownets2_flow = self.upsample3(flownets2_flow2 * self.div_flow)

        flow2_1 = self.conv7_1(flownets2_concat2)
        flow2_2 = self.conv7_2(flow2_1)
        flow2_3 = self.conv7_3(flow2_2)

        flow2_4 = self.conv8_1(flow2_3)
        flow2_5 = self.conv8_2(flow2_4)
        flow2_6 = self.conv8_3(flow2_5)

        flow2_result1 = self.conv7_down(flow2_3)
        flow2_result2 = self.conv8_down(flow2_6)

        fuse = torch.cat((flow2_result1, flow2_result2), dim=1)
        flow2_result = self.conv_final(fuse)

        return flow2_result