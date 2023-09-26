from pos_embed import get_2d_sincos_pos_embed

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import PatchEmbed, Block
import cv2
import numpy as np

class FlowHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, out_cha=2):   # mask时改成1
        super(FlowHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, 256, 3, padding=1)
        self.conv2 = nn.Conv2d(256, out_cha, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class UpdateBlock(nn.Module):
    def __init__(self, hidden_dim=128):
        super(UpdateBlock, self).__init__()
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)
        self.mask = nn.Sequential(
            nn.Conv2d(hidden_dim, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 16 * 16 * 9, 1, padding=0))

    def forward(self, imgf, coords1):
        #print(imgf.shape, coords1.shape)  # torch.Size([1, 14, 14, 512]) torch.Size([1, 2, 28, 28])
        #imgf = imgf.permute(0,3,1,2)
        mask = .25 * self.mask(imgf)  # scale mask to balence gradients
        dflow = self.flow_head(imgf)
        coords1 = coords1 + dflow

        return mask, coords1
        
        
class MUpdateBlock(nn.Module):
    def __init__(self, hidden_dim=128):
        super(MUpdateBlock, self).__init__()
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256, out_cha=1)
        self.mask = nn.Sequential(
            nn.Conv2d(hidden_dim, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 16 * 16 * 9, 1, padding=0))

    def forward(self, imgf):
    
        mask = .25 * self.mask(imgf)
        dflow = self.flow_head(imgf)

        return mask, dflow

        
class SimFIR(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=256, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()
        
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        # self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.enpred = nn.Linear(embed_dim, patch_size**2 * 2, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.update_block = UpdateBlock(patch_size ** 2)
        self.Mupdate_block = MUpdateBlock(patch_size ** 2)

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def upsample_flow(self, flow, mask):
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 16, 16, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(16 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)

        return up_flow.reshape(N, 2, 16 * H, 16 * W)
        
        
    def Mupsample_flow(self, flow, mask):
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 16, 16, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(16 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 1, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)

        return up_flow.reshape(N, 1, 16 * H, 16 * W)
        
    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))


        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def initialize_flow(self, img):
        N, C, H, W = img.shape
        coodslar = coords_grid(N, H, W).to(img.device)
        coords0 = coords_grid(N, H // 16, W // 16).to(img.device)
        coords1 = coords_grid(N, H // 16, W // 16).to(img.device)

        return coodslar, coords0, coords1
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):   #imgs is flow
        """
        imgs: (N, 2, H, W)
        x: (N, L, patch_size**2 *2)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 2, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 2))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *2)
        imgs: (N, 2, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 2))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 2, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, img):
        # embed patches
        x = self.patch_embed(img)
        
        # add pos embed w/o cls token
        x = x + self.pos_embed[:, :, :]
        
        for blk in self.blocks:
            x = blk(x)
            
        out_pos = x.reshape(x.shape[0], 16, 16, 256).permute(0, 3, 1, 2)
        
        coodslar, coords0, coords1 = self.initialize_flow(img)
        coords1 = coords1.detach()
        
        mask, coords1 = self.update_block(out_pos, coords1)
        flow_up = self.upsample_flow(coords1 - coords0, mask)
        bm_up = coodslar + flow_up
        
        # mask
        Mmask, Mcoords1 = self.Mupdate_block(out_pos)
        M_up = self.Mupsample_flow(Mcoords1, Mmask)

        return bm_up, torch.sigmoid(M_up)
        

    def forward_loss(self, imgs, pred):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)

        return loss

    def forward(self, imgs):
        pred = self.forward_encoder(imgs)
        
        return pred


def coords_grid(batch, ht, wd):
    coords = torch.meshgrid(torch.arange(ht), torch.arange(wd))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)
      
