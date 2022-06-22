import torch
import numpy as np
from scipy.stats import hmean


device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Evaluator:

    def __init__(self, dset, model):

        self.dset = dset
        pairs = [(dset.attr2idx[attr], dset.obj2idx[obj]) for attr, obj in dset.pairs]
        self.train_pairs = [(dset.attr2idx[attr], dset.obj2idx[obj]) for attr, obj in dset.train_pairs]
        self.pairs = torch.LongTensor(pairs)

        if dset.phase == 'train':
            print('Evaluating with train pairs')
            test_pair_set = set(dset.train_pairs)
            test_pair_gt = set(dset.train_pairs)
        elif dset.phase == 'val':
            print('Evaluating with validation pairs')
            test_pair_set = set(dset.val_pairs + dset.train_pairs)
            test_pair_gt = set(dset.val_pairs)
        else:
            print('Evaluating with test pairs')
            test_pair_set = set(dset.test_pairs + dset.train_pairs)
            test_pair_gt = set(dset.test_pairs)

        self.test_pair_dict = [(dset.attr2idx[attr], dset.obj2idx[obj]) for attr, obj in test_pair_gt]
        self.test_pair_dict = dict.fromkeys(self.test_pair_dict, 0)

       
        for attr, obj in test_pair_gt:
            pair_val = dset.pair2idx[(attr,obj)]
            key = (dset.attr2idx[attr], dset.obj2idx[obj])
            self.test_pair_dict[key] = [pair_val, 0, 0]

        if dset.open_world:
            masks = [1 for _ in dset.pairs]
        else:
            masks = [1 if pair in test_pair_set else 0 for pair in dset.pairs]

        self.closed_mask = torch.BoolTensor(masks)
        seen_pair_set = set(dset.train_pairs)
        mask = [1 if pair in seen_pair_set  else 0 for pair in dset.pairs]
        self.seen_mask = torch.BoolTensor(mask)

        oracle_obj_mask = []
        for _obj in dset.objs:
            mask = [1  if _obj == obj else 0 for attr, obj in dset.pairs]
            oracle_obj_mask.append(torch.BoolTensor(mask))
        self.oracle_obj_mask = torch.stack(oracle_obj_mask, 0)

        self.score_model = self.score_manifold_model

    # Generate mask for each settings, mask scores, and get prediction labels
    def generate_predictions(self, scores, obj_truth, bias = 0.0, topk = 5): # (Batch, #pairs)
        '''
        Inputs
            scores: Output scores
            obj_truth: Ground truth object
        Returns
            results: dict of results in 3 settings
        '''
        def get_pred_from_scores(_scores, topk):
            '''
            Given list of scores, returns top 10 attr and obj predictions
            Check later
            '''
            _, pair_pred = _scores.topk(topk, dim = 1) #sort returns indices of k largest values
            pair_pred = pair_pred.contiguous().view(-1)
            attr_pred, obj_pred = self.pairs[pair_pred][:, 0].view(-1, topk), \
                self.pairs[pair_pred][:, 1].view(-1, topk)
            return (attr_pred, obj_pred)

        results = {}
        orig_scores = scores.clone()
        mask = self.seen_mask.repeat(scores.shape[0],1) # Repeat mask along pairs dimension
        scores[~mask] += bias # Add bias to test pairs

        # Unbiased setting
        
        # Open world setting --no mask, all pairs of the dataset
        results.update({'open': get_pred_from_scores(scores, topk)})
        results.update({'unbiased_open': get_pred_from_scores(orig_scores, topk)})
        # Closed world setting - set the score for all Non test pairs to -1e10, 
        # this excludes the pairs from set not in evaluation
        mask = self.closed_mask.repeat(scores.shape[0], 1)
        closed_scores = scores.clone()
        closed_scores[~mask] = -1e10 
        closed_orig_scores = orig_scores.clone()
        closed_orig_scores[~mask] = -1e10
        results.update({'closed': get_pred_from_scores(closed_scores, topk)})
        results.update({'unbiased_closed': get_pred_from_scores(closed_orig_scores, topk)})

        # Object_oracle setting - set the score to -1e10 for all pairs where the true object does Not participate, can also use the closed score
        mask = self.oracle_obj_mask[obj_truth]
        oracle_obj_scores = scores.clone()
        oracle_obj_scores[~mask] = -1e10
        oracle_obj_scores_unbiased = orig_scores.clone()
        oracle_obj_scores_unbiased[~mask] = -1e10
        results.update({'object_oracle': get_pred_from_scores(oracle_obj_scores, 1)})
        results.update({'object_oracle_unbiased': get_pred_from_scores(oracle_obj_scores_unbiased, 1)})

        return results

    def score_clf_model(self, scores, obj_truth, topk = 5):
        '''
        Wrapper function to call generate_predictions for CLF models
        '''
        attr_pred, obj_pred = scores

        # Go to CPU
        attr_pred, obj_pred, obj_truth = attr_pred.to('cpu'), obj_pred.to('cpu'), obj_truth.to('cpu')

        # Gather scores (P(a), P(o)) for all relevant (a,o) pairs
        # Multiply P(a) * P(o) to get P(pair)
        attr_subset = attr_pred.index_select(1, self.pairs[:,0]) # Return only attributes that are in our pairs
        obj_subset = obj_pred.index_select(1, self.pairs[:, 1])
        scores = (attr_subset * obj_subset) # (Batch, #pairs)

        results = self.generate_predictions(scores, obj_truth)
        results['biased_scores'] = scores

        return results

    def score_manifold_model(self, scores, obj_truth, bias = 0.0, topk = 5):
        '''
        Wrapper function to call generate_predictions for manifold models
        '''
        # Go to CPU
        scores = {k: v.to('cpu') for k, v in scores.items()}
        obj_truth = obj_truth.to(device)

        # Gather scores for all relevant (a,o) pairs
        scores = torch.stack(
            [scores[(attr,obj)] for attr, obj in self.dset.pairs], 1
        ) # (Batch, #pairs)
        orig_scores = scores.clone()
        results = self.generate_predictions(scores, obj_truth, bias, topk)
        results['scores'] = orig_scores
        return results

    def score_fast_model(self, scores, obj_truth, bias = 0.0, topk = 5):
        '''
        Wrapper function to call generate_predictions for manifold models
        '''
        
        results = {}
        mask = self.seen_mask.repeat(scores.shape[0],1) # Repeat mask along pairs dimension
        scores[~mask] += bias # Add bias to test pairs

        mask = self.closed_mask.repeat(scores.shape[0], 1)
        closed_scores = scores.clone()
        closed_scores[~mask] = -1e10 

        _, pair_pred = closed_scores.topk(topk, dim = 1) #sort returns indices of k largest values
        pair_pred = pair_pred.contiguous().view(-1)
        attr_pred, obj_pred = self.pairs[pair_pred][:, 0].view(-1, topk), \
            self.pairs[pair_pred][:, 1].view(-1, topk)

        results.update({'closed': (attr_pred, obj_pred)})
        return results

    def evaluate_predictions(self, predictions, attr_truth, obj_truth, pair_truth, allpred, topk = 1):
        # Go to CPU
        attr_truth, obj_truth, pair_truth = attr_truth.to('cpu'), obj_truth.to('cpu'), pair_truth.to('cpu')

        pairs = list(
            zip(list(attr_truth.numpy()), list(obj_truth.numpy())))
        

        seen_ind, unseen_ind = [], []
        for i in range(len(attr_truth)):
            if pairs[i] in self.train_pairs:
                seen_ind.append(i)
            else:
                unseen_ind.append(i)

        
        seen_ind, unseen_ind = torch.LongTensor(seen_ind), torch.LongTensor(unseen_ind)
        def _process(_scores):
            # Top k pair accuracy
            # Attribute, object and pair
            attr_match = (attr_truth.unsqueeze(1).repeat(1, topk) == _scores[0][:, :topk])
            obj_match = (obj_truth.unsqueeze(1).repeat(1, topk) == _scores[1][:, :topk])

            # Match of object pair
            match = (attr_match * obj_match).any(1).float()
            attr_match = attr_match.any(1).float()
            obj_match = obj_match.any(1).float()
            # Match of seen and unseen pairs
            seen_match = match[seen_ind]
            unseen_match = match[unseen_ind]
            seen_score, unseen_score = torch.ones(512,5), torch.ones(512,5)

            return attr_match, obj_match, match, seen_match, unseen_match, \
            torch.Tensor(seen_score+unseen_score), torch.Tensor(seen_score), torch.Tensor(unseen_score)

        def _add_to_dict(_scores, type_name, stats):
            base = ['_attr_match', '_obj_match', '_match', '_seen_match', '_unseen_match', '_ca', '_seen_ca', '_unseen_ca']
            for val, name in zip(_scores, base):
                stats[type_name + name] = val

        ##################### Match in places where corrent object
        obj_oracle_match = (attr_truth == predictions['object_oracle'][0][:, 0]).float()  #object is already conditioned
        obj_oracle_match_unbiased = (attr_truth == predictions['object_oracle_unbiased'][0][:, 0]).float()

        stats = dict(obj_oracle_match = obj_oracle_match, obj_oracle_match_unbiased = obj_oracle_match_unbiased)

        #################### Closed world
        closed_scores = _process(predictions['closed'])
        unbiased_closed = _process(predictions['unbiased_closed'])
        _add_to_dict(closed_scores, 'closed', stats)
        _add_to_dict(unbiased_closed, 'closed_ub', stats)

        #################### Calculating AUC
        scores = predictions['scores']
        # getting score for each ground truth class
        correct_scores = scores[torch.arange(scores.shape[0]), pair_truth][unseen_ind]

        # Getting top predicted score for these unseen classes
        max_seen_scores = predictions['scores'][unseen_ind][:, self.seen_mask].topk(topk, dim=1)[0][:, topk - 1]

        # Getting difference between these scores
        unseen_score_diff = max_seen_scores - correct_scores

        # Getting matched classes at max bias for diff
        unseen_matches = stats['closed_unseen_match'].bool()
        correct_unseen_score_diff = unseen_score_diff[unseen_matches] - 1e-4

        # sorting these diffs
        correct_unseen_score_diff = torch.sort(correct_unseen_score_diff)[0]
        magic_binsize = 20
        # getting step size for these bias values
        bias_skip = max(len(correct_unseen_score_diff) // magic_binsize, 1)
        # Getting list
        biaslist = correct_unseen_score_diff[::bias_skip]

        seen_match_max = float(stats['closed_seen_match'].mean())
        unseen_match_max = float(stats['closed_unseen_match'].mean())
        seen_accuracy, unseen_accuracy = [], []

        # Go to CPU
        base_scores = {k: v.to('cpu') for k, v in allpred.items()}
        obj_truth = obj_truth.to('cpu')

        # Gather scores for all relevant (a,o) pairs
        base_scores = torch.stack(
            [allpred[(attr,obj)] for attr, obj in self.dset.pairs], 1
        ) # (Batch, #pairs)

        for bias in biaslist:
            scores = base_scores.clone()
            results = self.score_fast_model(scores, obj_truth, bias = bias, topk = topk)
            results = results['closed'] # we only need biased
            results = _process(results)
            seen_match = float(results[3].mean())
            unseen_match = float(results[4].mean())
            seen_accuracy.append(seen_match)
            unseen_accuracy.append(unseen_match)

        seen_accuracy.append(seen_match_max)
        unseen_accuracy.append(unseen_match_max)
        seen_accuracy, unseen_accuracy = np.array(seen_accuracy), np.array(unseen_accuracy)
        area = np.trapz(seen_accuracy, unseen_accuracy)

        for key in stats:
            stats[key] = float(stats[key].mean())

        harmonic_mean = hmean([seen_accuracy, unseen_accuracy], axis = 0)
        max_hm = np.max(harmonic_mean)
        idx = np.argmax(harmonic_mean)
        if idx == len(biaslist):
            bias_term = 1e3
        else:
            bias_term = biaslist[idx]
        stats['biasterm'] = float(bias_term)
        stats['best_unseen'] = np.max(unseen_accuracy)
        stats['best_seen'] = np.max(seen_accuracy)
        stats['AUC'] = area
        stats['hm_unseen'] = unseen_accuracy[idx]
        stats['hm_seen'] = seen_accuracy[idx]
        stats['best_hm'] = max_hm
        return stats