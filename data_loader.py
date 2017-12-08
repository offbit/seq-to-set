import numpy as np

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from spotlight.datasets.movielens import get_movielens_dataset
from spotlight.cross_validation import user_based_train_test_split


class InteractionsSampler(Dataset):

    def __init__(self, sequences, sequence_lengths,
                 template_size=70, query_size=10,
                 min_nb_interactions=50, num_negative=1000,perturb_prob=0.0):

        include = np.argwhere(sequence_lengths > min_nb_interactions).ravel()
        self.template_size = template_size
        self.query_size = query_size
        self.sequences = sequences[include]
        self.lengths = sequence_lengths[include]
        self.max_length = sequences.shape[-1]
        self.num_negative = num_negative
        self.num_items = int(np.max(sequences) + 3)  # plus 2 tokens {BOS, EOS}
        print('Loaded dataset of shape:{}\n'
              'Number of unique items: {}'.format(
                  self.sequences.shape, self.num_items))

        self.perturb_prob = perturb_prob

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (user interactions, positive query,
                    negative query, interaction length)
        """
        seq_length = self.lengths[index]
        sequence = self.sequences[index][-seq_length:]
        query_size = self.query_size
        user_interactions = np.zeros(self.template_size).astype(np.int64)

        interaction_length = self.template_size

        if seq_length > self.template_size + self.query_size:
            split = \
                np.random.randint(self.template_size, seq_length - query_size)
            user_interactions = \
                sequence[max(0, split - self.template_size):split]
        else:
            split = np.random.randint(1, seq_length - query_size)
            interaction_length = split
            user_interactions[-split:] = \
                sequence[max(0, split - self.template_size):split]
        
        # perturb each sequence elemetn with probability p: self.perturb_prob
        if self.perturb_prob > 0.0:
            chance = np.random.rand(interaction_length)
            random_ints = np.random.choice(
                self.num_items - 2, interaction_length, replace=False)
            for pos, coin in enumerate(chance):
                if coin < self.perturb_prob:
                    user_interactions[-pos] = random_ints[pos]
        # add BOS / EOS
        positive = [self.num_items - 2] + \
            list(sequence[split:split + query_size]) + [self.num_items - 1]
        positive = np.array(positive)
        
        # negative = list(np.random.choice(
        #  self.num_items - 2, query_size*2, replace=False))
        # add BOS/ EOS
        negative_ = np.setdiff1d(np.arange(self.num_items), np.array(self.sequences[index]))

        negative = np.random.choice(negative_, self.num_negative, replace=False)
        # negative = np.array([self.num_items - 2] + negative + [self.num_items - 1])
        
        
        user_interactions = torch.from_numpy(user_interactions.astype(np.int64))
        positive = torch.from_numpy(positive.astype(np.int64))
        negative = torch.from_numpy(negative.astype(np.int64))

        return user_interactions, positive, negative, interaction_length

    def __len__(self):
        return len(self.sequences)

def sort_batch(data, seq_len):                                                  
    """ 
    Sort the data (B, T, D) and sequence lengths                            
    """                                                                         
    sorted_seq_len, sorted_idx = seq_len.sort(0, descending=True)               
    sorted_data = data[sorted_idx]                                              
    return sorted_data, sorted_seq_len, sorted_idx                              

def read_dataset():
    max_sequence_length = 200
    min_sequence_length = 20
    step_size = 200
    random_state = np.random.RandomState(100)
    dataset = get_movielens_dataset('1M')

    train, rest = user_based_train_test_split(dataset,
                                              random_state=random_state)
    test, validation = user_based_train_test_split(rest,
                                                   test_percentage=0.5,
                                                   random_state=random_state)
    train = train.to_sequence(max_sequence_length=max_sequence_length,
                              min_sequence_length=min_sequence_length,
                              step_size=step_size)
    test = test.to_sequence(max_sequence_length=max_sequence_length,
                            min_sequence_length=min_sequence_length,
                            step_size=step_size)
    validation = validation.to_sequence(
        max_sequence_length=max_sequence_length,
        min_sequence_length=min_sequence_length,
        step_size=step_size)

    return train, test, validation


if __name__ == '__main__':

    train, test, val = read_dataset()
    db = InteractionsSampler(train.sequences, train.sequence_lengths, perturb_prob=0.1)
    train_loader = DataLoader(db, batch_size=16, shuffle=True)
    iterator = iter(train_loader)

    user_interactions, positive , negative, interaction_len = iterator.next()
