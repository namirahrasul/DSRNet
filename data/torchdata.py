import bisect #maintains a list in sorted order without having sort after each insertion using bisection algorithm
import warnings #issue warning message


class Dataset(object): # all python classes inherit from object
    """An abstract class representing a Dataset.

    All other datasets should subclass it. All subclasses should override
    ``__len__``, that provides the size of the dataset, and ``__getitem__``,
    supporting integer indexing in range from 0 to len(self) exclusive.
    """
    #traditional overloading is not in python

    def __getitem__(self, index): #__getitem__ for indexing with [ ] (overrriding)
        raise NotImplementedError

    def __len__(self): # for overriding len()
        raise NotImplementedError #when you call len(obj), Python internally calls obj.__len__().

    def __add__(self, other):  #to override an binary + operator
        return ConcatDataset([self, other])

    def reset(self): #currently is a placeholder and does nothing
        return 


class ConcatDataset(Dataset): #inheriting from Dataset
    """
    Dataset to concatenate multiple datasets.
    Purpose: useful to assemble different existing datasets, possibly
    large-scale datasets as the concatenation operation is done in an
    on-the-fly manner.

    Arguments:
        datasets (sequence): List of datasets to be concatenated
    """

    @staticmethod
    def cumsum(sequence): #sequence: an ordered collection of items, where each item holds a relative position.Must be indexable and can be iterated. All sequences are iterables but reverse is not true e.g. sets   not indexable
        #sequence   are strings, lists, tuples, byte sequences, byte arrays and range objects  
        r, s = [], 0  #r:list to hold cumulative sum of iterated datasets in sequence
                      #s: cumulative sum of all datasets in sequence
        for e in sequence: #e: current dataset in sequence
            l = len(e)      # length of current dataset in sequence
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets):
        super(ConcatDataset, self).__init__()
        assert len(datasets) > 0, 'datasets should not be an empty iterable'
                                  #halts the program and provides a diagnostic message.
                                  #asserts are usually stripped out or disabled in production code, as they are meant for development and debugging
        self.datasets = list(datasets) #store list of datasets
        self.cumulative_sizes = self.cumsum(self.datasets) #store result of static method cumsum, list of cumulative length of previous datasets

    def __len__(self):
        return self.cumulative_sizes[-1] #length of concatenated datasets after cumsum

    def __getitem__(self, idx): #dynamically selecting the correct dataset and item within that dataset based on the given index in the self.datasets
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx) #bisect.bisect_right returns the index of the first element in the list that is greater than x (upper bound)
        # line determines which dataset in self.datasets the item at idx belongs to.
        if dataset_idx == 0: #idx belongs to first dataset
            sample_idx = idx #so idx and cumulative idx same
        else:#finds idx within the dataset
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]

    @property #property of the class. Can be accessed without ()
    def cummulative_sizes(self): # basically a getter for encapsulation
        warnings.warn("cummulative_sizes attribute is renamed to " #basically if user uses wrong function
                      "cumulative_sizes", DeprecationWarning, stacklevel=2) #If the stacklevel parameter is not set, the warning appears to originate from within the cummulative_sizes method itself.
         #By setting stacklevel=2, the warning will appear to originate from the caller of the cummulative_sizes method.
        return self.cumulative_sizes 