from typing import List, Tuple


"""
NOTES (TO DO): not finished designing for smoothing, phi initialization, and viterbi algorithm
ideas: what if smoothing is applied as parameter of function?
    eg. _calculate_emission_matrix(self, ..., smoothing="log")
        _calculate_emission_matrix(self, ..., smoothing=None)
"""
class HMM:
    """
    HMM that implements Viterbi Algorithm

    Attributes:
    - a set of N states Q = q1, q2, . . . , qN (represents tags)
    - a sequence of T observations O = o1, o2, . . . , oT (represents words)
    - a transition probability matrix A = a11 . . . aij . . . aNN
    - a sequence of observation likelihoods B = bi(ot)
    - an initial probability distribution over states π = π1, π2, . . . , πN
    """
    def __init__(self):
        self.init_attributes()

    def init_attributes(self):
        self.N = 0 # total tags
        self.T = 0 # total words
        self.Q: List = [] # size N
        self.O: List = [] # size T
        self.A: List[List] = [] # transition_matrix with size N x N
        self.B = [] # emission_matrix
        self.phi = [] # size N

    def train(self, train_data: List[List[Tuple[str, str]]]):
        """
        Input: 
            train_data:
                list of pairs (word & tag) for each sentence
                eg.
                [[('The', 'DET'), ('Fulton', 'NOUN'), ...],
                [],
                []]

                meaning:
                - train_data[0] = List of pairs in the 1st sentence
                - train_data[0][0] = 1st pair of word & tag of the 1st sentence
        """
        self.init_attributes()

        words = set()
        tags = set()
        for sent in train_data:
            for word, tag in sent:
                words.add(word)
                tags.add(tag)
        
        self.N = len(tags)
        self.T = len(words)
        self.Q = list(tags)
        self.O = list(words)

        # to do
        self._calculate_emission_matrix()
        self._calculate_transition_matrix()
        pass
        

    def evaluate(self, gold_data: List[List[Tuple[str, str]]]):
        """
        Input: list of pairs (word & tag) for each sentence
            read the train function's input for further explanation
        
        Task:
            - strip the tags from the gold data, retag it using this HMM tagger (use predict function)
            - show / return the evaluation score (eg. accuracy)
        """
        pass

    def predict(self, sent: List[str]):
        """
        Input: List of words in a sentence
        Output: List of tags

        Notes: 
        - applying Viterbi algorithm
        - don't forget to handle the UNK token
        """
        pass
    
    def get_transition_matrix(self):
        return self.A
    
    def get_emission_matrix(self):
        return self.B
    
    def _calculate_emission_matrix(self):
        """
        Description: How likely a word 'bird' will be a noun
        Size: total tags x total words N x T (number of rows × number of columns)

            for example:
            
             | the | aged | bottle | flies | fast
         ADJ |     |      |        |       |
         ADV |     |      |        |       |
            ...

        """
        pass

    def _calculate_transition_matrix(self):
        """
        Description: How likely is noun is followed by a verb, etc
        Size: total tags (+1 token *) x total tags (+1 token STOP)
                (N + 1) x (N + 1)
                (number of rows × number of columns)

            for example:
            
             | ADJ | ADV | VERB | NOUN | STOP
          *  |     |     |      |      |
         ADJ |     |     |      |      |
         ADV |     |     |      |      |
            ...

            notes:
            - * means the start-token
            - STOP means the end-token
        """
        pass



