import os
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as tdata
import torchvision.models as tmodels
import torchvision.transforms as transforms
import tqdm

from PIL import Image


class ImageLoader:
    def __init__(self, root):
        self.img_dir = root

    def __call__(self, img):
        file = f'{self.img_dir}/{img}'
        img = Image.open(file).convert('RGB')
        return img


def imagenet_transform(phase):
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    if phase == 'train':
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
       
    elif phase == 'test' or phase == 'val':
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    return transform


def imagenet_transform_zappos(phase, cfg):
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    if phase == 'train':
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    elif phase == 'test' or phase == 'val':
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    return transform


class CompositionDataset(tdata.Dataset):
    def __init__(
        self,
        phase,
        split='compositional-split',
        open_world=False,
        cfg=None
    ):
        self.phase = phase
        self.cfg = cfg
        self.split = split
        self.open_world = open_world

        if 'ut-zap50k' in cfg.DATASET.name:
            self.transform = imagenet_transform_zappos(phase, cfg)
        else:
            self.transform = imagenet_transform(phase)
        self.loader = ImageLoader(f'{cfg.DATASET.root_dir}/images')
        
        self.attrs, self.objs, self.pairs, \
            self.train_pairs, self.val_pairs, \
            self.test_pairs = self.parse_split()

        self.train_data, self.val_data, self.test_data = self.get_split_info()
        if self.phase == 'train':
            self.data = self.train_data
        elif self.phase == 'val':
            self.data = self.val_data
        else:
            self.data = self.test_data

        self.obj2idx = {obj: idx for idx, obj in enumerate(self.objs)}
        self.attr2idx = {attr: idx for idx, attr in enumerate(self.attrs)}
        self.pair2idx = {pair: idx for idx, pair in enumerate(self.pairs)}
        self.train_pair2idx = {pair: idx for idx, pair in enumerate(self.train_pairs)}

        print('# train pairs: %d | # val pairs: %d | # test pairs: %d' % (len(
            self.train_pairs), len(self.val_pairs), len(self.test_pairs)))
        print('# train images: %d | # val images: %d | # test images: %d' %
              (len(self.train_data), len(self.val_data), len(self.test_data)))

        if cfg.TRAIN.use_precomputed_features:
            feat_file = f'{cfg.DATASET.root_dir}/features.t7'
            feat_avgpool = True
            if not os.path.exists(feat_file):
                with torch.no_grad():
                    self.generate_features(feat_file, feat_avgpool)

            activation_data = torch.load(feat_file)
            self.activations = dict(
                zip(activation_data['files'], activation_data['features']))
            self.feat_dim = activation_data['features'].size(1)

            print('%d activations loaded' % (len(self.activations)))

        # Affordance.
        self.attr_affordance = {} # -> contains objects compatible with an attribute.
        for _attr in self.attrs:
            candidates = [
                obj
                for (_, attr, obj) in self.train_data
                if attr == _attr
            ]
            self.attr_affordance[_attr] = sorted(list(set(candidates)))
            if len(self.attr_affordance[_attr]) <= 1:
                print(f'{_attr} is associated with <= 1 object: {self.attr_affordance[_attr]}')

        # Images that contain an object.
        self.image_with_obj = {}
        for i, instance in enumerate(self.train_data):
            obj = instance[2]
            if obj not in self.image_with_obj:
                self.image_with_obj[obj] = []
            self.image_with_obj[obj].append(i)
        
        # Images that contain an attribute.
        self.image_with_attr = {}
        for i, instance in enumerate(self.train_data):
            attr = instance[1]
            if attr not in self.image_with_attr:
                self.image_with_attr[attr] = []
            self.image_with_attr[attr].append(i)

        # Images that contain a pair.
        self.image_with_pair = {}
        for i, instance in enumerate(self.train_data):
            attr, obj = instance[1], instance[2]
            if (attr, obj) not in self.image_with_pair:
                self.image_with_pair[(attr, obj)] = []
            self.image_with_pair[(attr, obj)].append(i)
        
        if cfg.MODEL.use_composed_pair_loss:
            # with open('unseen_pairs/'+cfg.DATASET.name+'_unseen_pairs.txt', 'r') as f:
            #     self.unseen_pairs = [tuple(l.strip().split()) for l in f.readlines()]
            unseen_pairs = set()
            for pair in self.val_pairs + self.test_pairs:
                if pair not in self.train_pair2idx:
                    unseen_pairs.add(pair)
            self.unseen_pairs = list(unseen_pairs)
            self.unseen_pair2idx = {pair: idx for idx, pair in enumerate(self.unseen_pairs)}
            
    def get_split_info(self):
        data = torch.load(f'{self.cfg.DATASET.root_dir}/metadata_{self.split}.t7')
        train_data, val_data, test_data = [], [], []

        for instance in data:
            image, attr, obj, settype = \
                instance['image'], instance['attr'], instance['obj'], instance['set']
            if attr == 'NA' or (attr, obj) not in self.pairs or settype == 'NA':
                continue
            data_i = [image, attr, obj]
            if settype == 'train':
                train_data.append(data_i)
            elif settype == 'val':
                val_data.append(data_i)
            else:
                test_data.append(data_i)

        return train_data, val_data, test_data

    def parse_split(self):
        def parse_pairs(pair_list):
            with open(pair_list, 'r') as f:
                pairs = f.read().strip().split('\n')
                if self.cfg.DATASET.name == 'vaw-czsl':
                    pairs = [t.split('+') for t in pairs]
                else:
                    pairs = [t.split() for t in pairs]
                pairs = list(map(tuple, pairs))
            attrs, objs = zip(*pairs)
            return attrs, objs, pairs
        tr_attrs, tr_objs, tr_pairs = parse_pairs(
            f'{self.cfg.DATASET.root_dir}/{self.split}/train_pairs.txt')
        vl_attrs, vl_objs, vl_pairs = parse_pairs(
            f'{self.cfg.DATASET.root_dir}/{self.split}/val_pairs.txt')
        ts_attrs, ts_objs, ts_pairs = parse_pairs(
            f'{self.cfg.DATASET.root_dir}/{self.split}/test_pairs.txt')

        all_attrs, all_objs = sorted(
            list(set(tr_attrs + vl_attrs + ts_attrs))), sorted(
                list(set(tr_objs + vl_objs + ts_objs)))
        all_pairs = sorted(list(set(tr_pairs + vl_pairs + ts_pairs)))

        return all_attrs, all_objs, all_pairs, tr_pairs, vl_pairs, ts_pairs

    def __getitem__(self, index):
        image, attr, obj = self.data[index]
        if self.cfg.TRAIN.use_precomputed_features:
            img = self.activations[image]
        else:
            img = self.loader(image)
            img = self.transform(img)

        if self.phase == 'train':
            data = {
                'img': img,
                'attr': self.attr2idx[attr],
                'obj': self.obj2idx[obj],
                'pair': self.train_pair2idx[(attr, obj)],
                'img_name': self.data[index][0]
            }

            data['mask_task'] = 1 # Attribute task
            i2 = self.sample_same_attribute(attr, obj, with_different_obj=True)
            if i2 == -1:
                data['mask_task'] = 0
            img1, attr1, obj1_a = self.data[i2]

            if self.cfg.TRAIN.use_precomputed_features:
                img1 = self.activations[img1]
            else:
                img1 = self.loader(img1)
                img1 = self.transform(img1)

            data['img1_a'] = img1
            data['attr1_a'] = self.attr2idx[attr1]
            data['obj1_a'] = self.obj2idx[obj1_a]
            data['idx1_a'] = i2
            data['img1_name_a'] = self.data[i2][0]

            # Object task.
            i2 = self.sample_same_object(attr, obj, with_different_attr=True)
            img1, attr1_o, obj1 = self.data[i2]

            if self.cfg.TRAIN.use_precomputed_features:
                img1 = self.activations[img1]
            else:
                img1 = self.loader(img1)
                img1 = self.transform(img1)
            data['img1_o'] = img1
            data['attr1_o'] = self.attr2idx[attr1_o]
            data['obj1_o'] = self.obj2idx[obj1]
            data['idx1_o'] = i2
            data['img1_name_o'] = self.data[i2][0]

            if self.cfg.MODEL.use_composed_pair_loss:
                if (attr1_o, obj1_a) in self.unseen_pair2idx:
                    data['composed_unseen_pair'] = self.unseen_pair2idx[(attr1_o, obj1_a)]
                    data['composed_seen_pair'] = 2000
                elif (attr1_o, obj1_a) in self.train_pair2idx:
                    data['composed_seen_pair'] = self.train_pair2idx[(attr1_o, obj1_a)]
                    data['composed_unseen_pair'] = 2000
                else:
                    data['composed_unseen_pair'] = 2000
                    data['composed_seen_pair'] = 2000

        else:
            # Testing mode.
            data = {
                'img': img,
                'attr': self.attr2idx[attr],
                'obj': self.obj2idx[obj],
                'pair': self.pair2idx[(attr, obj)],
            }
        return data

    def __len__(self):
        return len(self.data)

    def sample_same_attribute(self, attr, obj, with_different_obj=True):
        if with_different_obj:
            if len(self.attr_affordance[attr]) == 1:
                return -1
            i2 = np.random.choice(self.image_with_attr[attr])
            img1, attr1, obj1 = self.data[i2]
            while obj1 == obj:
                i2 = np.random.choice(self.image_with_attr[attr])
                img1, attr1, obj1 = self.data[i2]
            assert obj1 != obj
        else:
            i2 = np.random.choice(self.image_with_attr[attr])
        return i2

    def sample_same_object(self, attr, obj, with_different_attr=True):
        i2 = np.random.choice(self.image_with_obj[obj])
        if with_different_attr:
            img1, attr1, obj1 = self.data[i2]
            while attr1 == attr:
                i2 = np.random.choice(self.image_with_obj[obj])
                img1, attr1, obj1 = self.data[i2]
        return i2

    def generate_features(self, out_file, feat_avgpool=True):
        data = self.train_data + self.val_data + self.test_data
        transform = imagenet_transform('test')
        feat_extractor = tmodels.resnet18(pretrained=True)
        feat_extractor.fc = nn.Sequential()
        feat_extractor.eval().cuda()

        image_feats = []
        image_files = []
        for chunk in tqdm.tqdm(
                chunks(data, 512), total=len(data) // 512):
            files, attrs, objs = zip(*chunk)
            imgs = list(map(self.loader, files))
            imgs = list(map(transform, imgs))
            imgs = torch.stack(imgs, 0).cuda()
            if feat_avgpool:
                feats = feat_extractor(imgs)
            else:
                feats = feat_extractor.conv1(imgs)
                feats = feat_extractor.bn1(feats)
                feats = feat_extractor.relu(feats)
                feats = feat_extractor.maxpool(feats)
                feats = feat_extractor.layer1(feats)
                feats = feat_extractor.layer2(feats)
                feats = feat_extractor.layer3(feats)
                feats = feat_extractor.layer4(feats)
                assert feats.shape[-3:] == (512, 7, 7), feats.shape
            image_feats.append(feats.data.cpu())
            image_files += files
        image_feats = torch.cat(image_feats, 0)
        print('features for %d images generated' % (len(image_files)))
        torch.save({'features': image_feats, 'files': image_files}, out_file)


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]