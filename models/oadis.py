import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import Backbone
from .basic_layers import MLP
from .word_embedding_utils import initialize_wordembedding_matrix

class OADIS(nn.Module):
    """Object-Attribute Compositional Learning from Image Pair.
    """
    def __init__(self, dset, cfg):
        super(OADIS, self).__init__()
        self.cfg = cfg

        self.num_attrs = len(dset.attrs)
        self.num_objs = len(dset.objs)
        self.pair2idx = dset.pair2idx

        # Set training pairs.
        train_attrs, train_objs = zip(*dset.train_pairs)
        train_attrs = [dset.attr2idx[attr] for attr in train_attrs]
        train_objs = [dset.obj2idx[obj] for obj in train_objs]
        self.train_attrs = torch.LongTensor(train_attrs).cuda()
        self.train_objs = torch.LongTensor(train_objs).cuda()

        unseen_pair_attrs, unseen_pair_objs = zip(*dset.unseen_pairs)
        unseen_pair_attrs = [dset.attr2idx[attr] for attr in unseen_pair_attrs]
        unseen_pair_objs = [dset.obj2idx[obj] for obj in unseen_pair_objs]
        self.unseen_pair_attrs = torch.LongTensor(unseen_pair_attrs).cuda()
        self.unseen_pair_objs = torch.LongTensor(unseen_pair_objs).cuda()

        # Dimension of the joint image-label embedding space.
        if '+' in cfg.MODEL.wordembs:
            self.emb_dim = cfg.MODEL.emb_dim*2
        else:
            self.emb_dim = cfg.MODEL.emb_dim

        # Setup layers for word embedding composer.
        self._setup_word_composer(dset, cfg)

        if not cfg.TRAIN.use_precomputed_features and not cfg.TRAIN.comb_features:
            self.feat_extractor = Backbone('resnet18')
            feat_dim = 512

        img_emb_modules = [
            nn.Conv2d(feat_dim, cfg.MODEL.img_emb_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(cfg.MODEL.img_emb_dim),
            nn.ReLU()
        ]
        feat_dim = cfg.MODEL.img_emb_dim

        if cfg.MODEL.img_emb_drop > 0:
            img_emb_modules += [
                nn.Dropout2d(cfg.MODEL.img_emb_drop)]

        self.img_embedder = nn.Sequential(*img_emb_modules)
        self.img_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.img_final = nn.Linear(feat_dim, self.emb_dim)
        self.classifier = CosineClassifier(temp=cfg.MODEL.cosine_cls_temp)

        self.image_pair_comparison = ImagePairComparison(
            cfg, self.num_attrs, self.num_objs, self.attr_embedder, self.obj_embedder,
            img_dim=feat_dim,
            emb_dim=self.emb_dim,
            word_dim=self.word_dim,
            lambda_attn=cfg.MODEL.lambda_attn,
            attn_normalized=cfg.MODEL.attn_normalized
        )

        self.pair_final = nn.Linear(self.emb_dim*2, self.emb_dim)

    def _setup_word_composer(self, dset, cfg):
        attr_wordemb, self.word_dim = \
            initialize_wordembedding_matrix(cfg.MODEL.wordembs, dset.attrs, cfg)
        obj_wordemb, _ = \
            initialize_wordembedding_matrix(cfg.MODEL.wordembs, dset.objs, cfg)

        self.attr_embedder = nn.Embedding(self.num_attrs, self.word_dim)
        self.obj_embedder = nn.Embedding(self.num_objs, self.word_dim)
        self.attr_embedder.weight.data.copy_(attr_wordemb)
        self.obj_embedder.weight.data.copy_(obj_wordemb)

        # Dimension of the joint image-label embedding space.
        if '+' in cfg.MODEL.wordembs:
            emb_dim = cfg.MODEL.emb_dim*2
        else:
            emb_dim = cfg.MODEL.emb_dim

        self.wordemb_compose = cfg.MODEL.wordemb_compose
        if cfg.MODEL.wordemb_compose == 'linear':
            # Linear composer.
            self.compose = nn.Sequential(
                nn.Dropout(cfg.MODEL.wordemb_compose_dropout),
                nn.Linear(self.word_dim*2, self.emb_dim)
            )
        elif cfg.MODEL.wordemb_compose == 'obj-conditioned':
            # Composer conditioned on object.
            self.object_code = nn.Sequential(
                nn.Linear(self.word_dim, 600),
                nn.ReLU(True)
            )
            self.attribute_code = nn.Sequential(
                nn.Linear(self.word_dim, 600),
                nn.ReLU(True)
            )
            self.attribute_code_fc = nn.Sequential(
                nn.Linear(600, 600),
                nn.ReLU(True),
            )
            self.compose = MLP(
                self.word_dim + 600, 600, emb_dim, 2, batchnorm=False,
                drop_input=cfg.MODEL.wordemb_compose_dropout
            )
        elif cfg.MODEL.wordemb_compose == 'obj-conditioned-vaw':
            # Composer conditioned on object.
            self.object_code = nn.Sequential(
                nn.Linear(self.word_dim, 300),
                nn.ReLU(True)
            )
            self.attribute_code = nn.Sequential(
                nn.Linear(self.word_dim, 300),
                nn.ReLU(True)
            )
            self.compose = nn.Sequential(
                nn.Dropout(cfg.MODEL.wordemb_compose_dropout),
                nn.Linear(self.word_dim + 300, 300)
            )

    def compose_word_embeddings(self, mode='train'):
        if mode == 'train':
            attr_emb = self.attr_embedder(self.train_attrs) # [n_pairs, word_dim].
            obj_emb = self.obj_embedder(self.train_objs) # # [n_pairs, word_dim].
        elif mode == 'all':
            attr_emb = self.attr_embedder(self.all_attrs) # [n_pairs, word_dim].
            obj_emb = self.obj_embedder(self.all_objs)
        elif mode == 'unseen':
            attr_emb = self.attr_embedder(self.unseen_pair_attrs) # [n_pairs, word_dim].
            obj_emb = self.obj_embedder(self.unseen_pair_objs)
        else:
            # Expect val_attrs and val_objs are already set (using _set_val_pairs()).
            attr_emb = self.attr_embedder(self.val_attrs) # [n_pairs, word_dim].
            obj_emb = self.obj_embedder(self.val_objs) # # [n_pairs, word_dim].

        if 'obj-conditioned' in self.cfg.MODEL.wordemb_compose:
            object_c = self.object_code(obj_emb) # [n_pairs, 1024].
            attribute_c = self.attribute_code(attr_emb) # [n_pairs, 1024].
            if 'vaw' in self.cfg.MODEL.wordemb_compose:
                attribute_c = object_c * attribute_c
            else:
                attribute_c = self.attribute_code_fc(object_c * attribute_c)
            concept_emb = torch.cat((obj_emb, attribute_c), dim=-1) # [n_pairs, word_dim + 1024].
        else:
            concept_emb = torch.cat((obj_emb, attr_emb), dim=-1)
        concept_emb = self.compose(concept_emb) # [n_pairs, emb_dim].

        return concept_emb

    def train_forward(self, batch):
        img1 = batch['img']
        img2_a = batch['img1_a'] # Image that shares the same attribute
        img2_o = batch['img1_o'] # Image that shares the same object

        # Labels of 1st image.
        attr_labels = batch['attr']
        obj_labels = batch['obj']
        pair_labels = batch['pair']

        composed_unseen_pair = batch['composed_unseen_pair']
        composed_seen_pair = batch['composed_seen_pair']

        mask_task = batch['mask_task']
        bs = img1.shape[0]

        concept = self.compose_word_embeddings(mode='train') # (n_pairs, emb_dim)

        if not self.cfg.TRAIN.use_precomputed_features and not self.cfg.TRAIN.comb_features:
            img1 = self.feat_extractor(img1)[0]
            img2_a = self.feat_extractor(img2_a)[0]
            img2_o = self.feat_extractor(img2_o)[0]
        
        if not self.cfg.TRAIN.use_precomputed_features and self.cfg.TRAIN.comb_features:
            img1 = self.feat_extractor(img1)
            img2_a = self.feat_extractor(img2_a)
            img2_o = self.feat_extractor(img2_o)
        
        h, w = img1.shape[2:]
        img1 = self.img_embedder(img1).view(bs, -1, h*w)
        img2_a = self.img_embedder(img2_a).view(bs, -1, h*w)
        img2_o = self.img_embedder(img2_o).view(bs, -1, h*w)

        aux_loss = self.image_pair_comparison(
            img1, img2_a, img2_o, attr_labels, obj_labels, mask_task)

        img1 = self.img_avg_pool(img1.view(bs, -1, h, w)).squeeze()
        img1 = self.img_final(img1)
        
        pred = self.classifier(img1, concept)

        pair_loss = F.cross_entropy(pred, pair_labels)
        loss = pair_loss * self.cfg.MODEL.w_loss_main

        pred = torch.max(pred, dim=1)[1]
        attr_pred = self.train_attrs[pred]
        obj_pred = self.train_objs[pred]

        correct_attr = (attr_pred == attr_labels)
        correct_obj = (obj_pred == obj_labels)
        correct_pair = (pred == pair_labels)

        if self.cfg.MODEL.use_attr_loss:
            loss = loss + aux_loss['loss_attr'] * self.cfg.MODEL.w_loss_attr
        if self.cfg.MODEL.use_obj_loss:
            loss = loss + aux_loss['loss_obj'] * self.cfg.MODEL.w_loss_obj
        if self.cfg.MODEL.use_emb_pair_loss:
            pair_emb = self.pair_final(torch.cat((aux_loss['attr_feat2'], aux_loss['obj_feat2']), 1))
            mask = aux_loss['mask']
            pred = self.classifier(pair_emb, concept)
            emb_pair_loss = F.cross_entropy(pred, pair_labels[mask])
            loss = loss + emb_pair_loss * self.cfg.MODEL.emb_loss_main

        ### hallucinating unseen pairs
        if self.cfg.MODEL.use_composed_pair_loss:
            unseen_concept = self.compose_word_embeddings(mode='unseen')

            mask_unseen = aux_loss['mask'] & (composed_unseen_pair != 2000)
            if mask_unseen.sum() > 0:
                attr_emb = aux_loss['diff_a'][mask_unseen]
                obj_emb = aux_loss['diff_o'][mask_unseen]
                pair_unseen_emb = self.pair_final(torch.cat((attr_emb, obj_emb), 1))
                pred_unseen = self.classifier(pair_unseen_emb, unseen_concept)
                composed_unseen_loss = F.cross_entropy(pred_unseen, composed_unseen_pair[mask_unseen])    
                loss = loss + composed_unseen_loss * self.cfg.MODEL.unseen_loss_ratio
            else:
                composed_unseen_loss = torch.tensor([0.0], requires_grad=True)

            mask_seen = aux_loss['mask'] & (composed_seen_pair != 2000)
            if mask_seen.sum() > 0:
                attr_emb = aux_loss['diff_a'][mask_seen]
                obj_emb = aux_loss['diff_o'][mask_seen]
                pair_seen_emb = self.pair_final(torch.cat((attr_emb, obj_emb), 1))
                pred_seen = self.classifier(pair_seen_emb, concept)
                composed_seen_loss = F.cross_entropy(pred_seen, composed_seen_pair[mask_seen])    
                loss = loss + composed_seen_loss * self.cfg.MODEL.seen_loss_ratio
            else:
                composed_seen_loss = torch.tensor([0.0], requires_grad=True)
           
        out = {
            'loss_total': loss,
            'acc_attr': torch.div(correct_attr.sum(),float(bs)), 
            'acc_obj': torch.div(correct_obj.sum(),float(bs)), 
            'acc_pair': torch.div(correct_pair.sum(),float(bs)) 
        }

        if self.cfg.MODEL.use_attr_loss:
            out['loss_aux_attr'] = aux_loss['loss_attr']
            out['acc_aux_attr'] = aux_loss['acc_attr']

        if self.cfg.MODEL.use_obj_loss:
            out['loss_aux_obj'] = aux_loss['loss_obj']
            out['acc_aux_obj'] = aux_loss['acc_obj']

        if self.cfg.MODEL.use_emb_pair_loss:
            out['emb_loss'] = emb_pair_loss

        if self.cfg.MODEL.use_composed_pair_loss:
            # out['unseen_loss'] = unseen_loss
            out['composed_unseen_loss'] = composed_unseen_loss
            out['composed_seen_loss'] = composed_seen_loss

        return out

    def val_forward(self, batch):
        img = batch['img']
        bs = img.shape[0]

        concept = self.compose_word_embeddings(mode='val') # [n_pairs, emb_dim].
  
        if not self.cfg.TRAIN.use_precomputed_features and not self.cfg.TRAIN.comb_features:
            img = self.feat_extractor(img)[0]
        if not self.cfg.TRAIN.use_precomputed_features and self.cfg.TRAIN.comb_features:
            img = self.feat_extractor(img)
        h, w = img.shape[2:]
        img = self.img_embedder(img).view(bs, -1, h*w)
        img = self.img_avg_pool(img.view(bs, -1, h, w)).squeeze()
        img = self.img_final(img)

        pred = self.classifier(img, concept, scale=False)

        out = {}
        out['pred'] = pred

        out['scores'] = {}
        for _, pair in enumerate(self.val_pairs):
            out['scores'][pair] = pred[:,self.pair2idx[pair]]

        return out

    def forward(self, x):
        if self.training:
            out = self.train_forward(x)
        else:
            with torch.no_grad():
                out = self.val_forward(x)
        return out


class ImagePairComparison(nn.Module):
    """Cross attention module to find difference/similarity between two images.
    """
    def __init__(
        self,
        cfg,
        num_attrs,
        num_objs,
        attr_embedder,
        obj_embedder,
        img_dim=300,
        emb_dim=300,
        word_dim=300,
        lambda_attn=10,
        attn_normalized=True,
    ):
        super(ImagePairComparison, self).__init__()

        self.num_attrs = num_attrs
        self.num_objs = num_objs

        self.train_attrs = torch.LongTensor(list(range(self.num_attrs))).cuda()
        self.train_objs = torch.LongTensor(list(range(self.num_objs))).cuda()

        self.attr_embedder = attr_embedder
        self.obj_embedder = obj_embedder

        self.lambda_attn = lambda_attn
        self.attn_normalized = attn_normalized
        
        feat_dim = img_dim

        self.use_attr_loss = cfg.MODEL.use_attr_loss
        if self.use_attr_loss:
            self.sim_attr_embed = nn.Linear(feat_dim, emb_dim)
            if cfg.MODEL.wordemb_compose_dropout > 0:
                self.attr_mlp = nn.Sequential(
                    nn.Dropout(cfg.MODEL.wordemb_compose_dropout),
                    nn.Linear(word_dim, emb_dim)
                )
            else:
                self.attr_mlp = nn.Linear(word_dim, emb_dim)
            self.classify_attr = CosineClassifier(cfg.MODEL.cosine_cls_temp)

        self.use_obj_loss = cfg.MODEL.use_obj_loss
        if self.use_obj_loss:
            self.sim_obj_embed = nn.Linear(feat_dim, emb_dim)
            if cfg.MODEL.wordemb_compose_dropout > 0:
                self.obj_mlp = nn.Sequential(
                    nn.Dropout(cfg.MODEL.wordemb_compose_dropout),
                    nn.Linear(word_dim, emb_dim)
                )
            else:
                self.obj_mlp = nn.Linear(word_dim, emb_dim)
            self.classify_obj = CosineClassifier(cfg.MODEL.cosine_cls_temp)
        
    def func_attention(self, img1, img2):
        """
        img1: (bs, d, L)
        img2: (bs, d, L)
        """
        # Get attention
        # --> (bs, L, d)l
        img1T = torch.transpose(img1, 1, 2)

        # (bs, L, d)(bs, d, L)
        # --> (bs, L, L)
        if self.attn_normalized:
            relevance = torch.bmm(F.normalize(img1T, dim=2), F.normalize(img2, dim=1))
            non_relevance = -relevance
        else:
            relevance = torch.matmul(img1T, img2) / np.sqrt(2048)

        row_attn = F.softmax(relevance * self.lambda_attn, dim=2) # img1 -> img2 attention
        col_attn = F.softmax(relevance * self.lambda_attn, dim=1) # img2 -> img1 attention

        sim12 = row_attn.sum(1) # (bs, L) -> locations in img2 that are similar to many parts in img1
        sim21 = col_attn.sum(2) # (bs, L) -> locations in img1 that are similar to many parts in img2

        row_inv_attn = F.softmax(non_relevance * self.lambda_attn, dim=2)
        diff12 = row_inv_attn.sum(1) # (bs, L) -> locations in img2 that differ from most parts in img1

        # Normalize to get sum = 1.
        sim12 = sim12 / (sim12.sum(1, keepdim=True) + 1e-8)
        sim21 = sim21 / (sim21.sum(1, keepdim=True) + 1e-8)
        diff12 = diff12 / (diff12.sum(1, keepdim=True) + 1e-8)
        
        return sim12, sim21, diff12

    def forward_attn(self, image1, image2, fg1=None, fg2=None):
        sim12, sim21, diff12 = self.func_attention(image1, image2)

        # (bs, emb_dim, L) (bs, 1, L) -> (bs, emb_dim)
        sim_vec1 = (image1 * sim21.unsqueeze(1)).sum(2)
        sim_vec2 = (image2 * sim12.unsqueeze(1)).sum(2)

        diff_vec2 = (image2 * diff12.unsqueeze(1)).sum(2)

        return sim_vec1, sim_vec2, sim21, sim12, diff_vec2

    def forward(self, img1, img2_a, img2_o, attr1, obj1, mask_task):
        """
        """
        sim_vec1_a, sim_vec2_a, sim21_a, sim12_a, diff_o = self.forward_attn(img1, img2_a)
        sim_vec1_o, sim_vec2_o, sim21_o, sim12_o, diff_a  = self.forward_attn(img1, img2_o)

        mask = (mask_task == 1)

        out = {
            'mask': mask,
            'diff_a': self.sim_attr_embed(diff_a),
            'diff_o': self.sim_obj_embed(diff_o)
        }

        if self.use_attr_loss:
            attr_emb = self.attr_embedder(self.train_attrs)
            attr_weight = self.attr_mlp(attr_emb)

            attr_feat1 = self.sim_attr_embed(sim_vec1_a[mask])
            attr_pred1 = self.classify_attr(attr_feat1, attr_weight)
            attr_loss1 = F.cross_entropy(attr_pred1, attr1[mask])
            attr_pred1 = torch.max(attr_pred1, dim=1)[1]
            attr_pred1 = self.train_attrs[attr_pred1]
            correct_attr1 = (attr_pred1 == attr1[mask])

            attr_feat2 = self.sim_attr_embed(sim_vec2_a[mask])
            attr_pred2 = self.classify_attr(attr_feat2, attr_weight)
            attr_loss2 = F.cross_entropy(attr_pred2, attr1[mask])
            attr_pred2 = torch.max(attr_pred2, dim=1)[1]
            attr_pred2 = self.train_attrs[attr_pred2]
            correct_attr2 = (attr_pred2 == attr1[mask])

            out['loss_attr'] = (attr_loss1 + attr_loss2) / 2.0
            out['acc_attr'] = torch.div(torch.div(correct_attr1.sum().float(),mask.sum()) + \
                            torch.div(correct_attr2.sum().float(),mask.sum()), float(2))

            out['attr_feat1'] = attr_feat1
            out['attr_feat2'] = attr_feat2

        if self.use_obj_loss:
            obj_emb = self.obj_embedder(self.train_objs)
            obj_weight = self.obj_mlp(obj_emb)

            obj_feat1 = self.sim_obj_embed(sim_vec1_o[mask])
            obj_pred1 = self.classify_obj(obj_feat1, obj_weight)
            obj_loss1 = F.cross_entropy(obj_pred1, obj1[mask])
            obj_pred1 = torch.max(obj_pred1, dim=1)[1]
            obj_pred1 = self.train_objs[obj_pred1]
            correct_obj1 = (obj_pred1 == obj1[mask])

            obj_feat2 = self.sim_obj_embed(sim_vec2_o[mask])
            obj_pred2 = self.classify_obj(obj_feat2, obj_weight)
            obj_loss2 = F.cross_entropy(obj_pred2, obj1[mask])
            obj_pred2 = torch.max(obj_pred2, dim=1)[1]
            obj_pred2 = self.train_objs[obj_pred2]
            correct_obj2 = (obj_pred2 == obj1[mask])

            out['loss_obj'] = (obj_loss1 + obj_loss2) / 2.0
            out['acc_obj'] = torch.div(torch.div(correct_obj1.sum().float(),mask.sum()) + \
                              torch.div(correct_obj2.sum().float(),mask.sum()),float(2))

            out['obj_feat1'] = obj_feat1
            out['obj_feat2'] = obj_feat2

        return out


class CosineClassifier(nn.Module):
    def __init__(self, temp=0.05):
        super(CosineClassifier, self).__init__()
        self.temp = temp

    def forward(self, img, concept, scale=True):
        """
        img: (bs, emb_dim)
        concept: (n_class, emb_dim)
        """
        img_norm = F.normalize(img, dim=-1)
        concept_norm = F.normalize(concept, dim=-1)
        pred = torch.matmul(img_norm, concept_norm.transpose(0, 1))
        if scale:
            pred = pred / self.temp
        return pred