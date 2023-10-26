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
        self.transition_matrix = {} # transition_matrix
        self.emission_matrix = {} # emission_matrix
        self.phi = [] # size N

    def train(self, train_data: List[Tuple[str, str]], tag_list: set[str]):
        """
        Input: 
            train_data:
                list of pairs (word & tag)
                eg.
                [('The', 'DET'), ('Fulton', 'NOUN'), ...]

                meaning:
                - train_data[0] = 1st pair of word & tag
            
            tag_list:
                a set of all the tags present in the data set 
        """
        self.init_attributes()

        # Calculating both the Tranisition and Emission Matrix
        self.emission_matrix = self._calculate_emission_matrix(train_data, tag_list)
        self.transition_matrix = self._calculate_transition_matrix(train_data, tag_list)

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
        return self.transition_matrix
    
    def get_emission_matrix(self):
        return self.emission_matrix
    
    def _calculate_emission_matrix(self, train_data: List[Tuple[str, str]], tag_list: set[str]):
        """
        Calculates the emission matrix, which represents the conditional probabilities of words given tags.

        Input: 
            train_data:
                A list of pairs (word & tag).
            
            tag_list:
                A set of all the tags.

        Output: 
            emission_matrix:
                - Type: Dictionary
                - Structure : {'tags': {'words': emission probability} }
                
                For example: 
                    {
                        'Noun': {
                            'apple': 0.5,
                            'banana': 0.3,
                            'tree': 0.2
                        },
                        'Verb': {
                            'run': 0.6,
                            'jump': 0.4
                        }
                    }
        """
        tag_word_counts = {}  # Dictionary to store word counts for each tag
        tag_counts = {}  # Dictionary to store tag counts
        emission_matrix = {}  # Dictionary to store the emission matrix

        for word, tag in train_data:
            
            # Normalize word by converting it to lowercase
            word = word.lower()
            
            if tag in tag_word_counts:
                # If the tag exists in tag_word_counts, update word count
                tag_word_counts[tag][word] = tag_word_counts[tag].get(word, 0) + 1
            else:
                # If the tag is encountered for the first time, create a new entry
                tag_word_counts[tag] = {word: 1}
            
            # Update the tag count
            tag_counts[tag] = tag_counts.get(tag, 0) + 1

        for tag in tag_list:
            temp_matrix = {}  # Dictionary for the current tag

            for word in tag_word_counts.get(tag, {}):
                word_count = tag_word_counts[tag][word]
                temp_matrix[word] = word_count / tag_counts[tag]

            # Store the tag's emission probabilities in the emission matrix
            emission_matrix[tag] = temp_matrix

        return emission_matrix

    def _calculate_transition_matrix(self, train_data: List[Tuple[str, str]], tag_list: set[str]):
        """
        Calculates the transition matrix, which represents the probabilities of a specific tag given a previous tag.
        
        Formula: Transition_Probability(Tag2 | Tag1)

        Input: 
            train_data:
                A list of pairs (word & tag).
            
            tag_list:
                A set of all the tags.

        Output: 
            transition_matrix:
                - Type: Dictionary
                - Structure : {'tag2': {'tag1': transition probability} }
                
                For example: 
                {
                    'Noun': {
                        'Noun': 0.4,
                        'Verb': 0.3,
                        'Adjective': 0.2,
                    },
                    'Verb': {
                        'Noun': 0.1,
                        'Verb': 0.5,
                        'Adjective': 0.4,
                    },
                }
                
                So, Noun following as Adjective would have a probability of 0.2, 
                >>>> transition_matrix['Noun']['Adjective']
        """
        # Initialize an empty dictionary to store the transition matrix
        transition_matrix = {}

        # Count occurrences of each tag in the training data
        tag_counts = {}
        for word, tag in train_data:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1

        # Calculate transition probabilities for each pair of tags (t1, t2)
        for t2 in tag_list:
            
            # Initialize a dictionary to store transition probabilities for t2 given all possible t1 tags
            transition_matrix[t2] = {}
            for t1 in tag_list:
                count_t2_t1 = 0

                # Count tag transitions from t1 to t2 in the training data
                for idx in range(len(train_data) - 1):
                    if train_data[idx][1] == t1 and train_data[idx + 1][1] == t2:
                        count_t2_t1 += 1

                # Calculate and store the transition probability P(t2|t1)
                transition_matrix[t2][t1] = count_t2_t1 / tag_counts[t1]

        # Return the computed transition matrix
        return transition_matrix
