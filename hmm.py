from typing import List, Tuple, Optional
import math
from const import UNK_TOKEN, START_TOKEN, STOP_TOKEN

"""
NOTES (TO DO): not finished viterbi algorithm and add token start & end for each sentences (in train & predict)
"""


class HMM:
    """
    HMM that implements Viterbi Algorithm

    Attributes (in theory):
    - a set of N states Q = q1, q2, . . . , qN (represents tags)
    - a sequence of T observations O = o1, o2, . . . , oT (represents words)
    - a transition probability matrix A = a11 . . . aij . . . aNN
    - a sequence of observation likelihoods B = bi(ot)
    - an initial probability distribution over states π = π1, π2, . . . , πN

    Attributes (in application):
    - transition probability matrix has size of 'tags found in training data' + 3 (added UNK_TOKEN, START_TOKEN, END_TOKEN)
    - initial probability distribution over states (phi) also has size of 'tags found in training data' + 3
    - emission matrix / sequence of observation likelihoods B has size of 'tags found in training data' + 1 (added UNK_TOKEN)
    - use_log_prob is used if the value in emission / transition matrix is calculated using log or not.
        probability is a decimal number, when it is not a log value and used in multiplications, it may lead to underflow
    - apply_smoothing_in_emission_matrix is used to give smoothing to the emission matrix or not.
        smoothing used to give unseen events a small amount of probability.
    - self.apply_smoothing_in_transition_matrix is used to give smoothing to the transition matrix or not.
        smoothing used to give unseen events a small amount of probability.
    """

    def __init__(self):
        self._init_attributes()

    def _init_attributes(self):
        self.N = 0  # total tags
        self.T = 0  # total words
        self.Q: List = []  # list tags size N
        self.O: List = []  # list words size T
        self.transition_matrix = {}  # transition_matrix
        self.emission_matrix = {}  # emission_matrix
        self.smoothing_factor = 0.0  # default value
        self.phi = []  # size N

        # other attributes for tuning the models
        self.use_log_prob = False
        self.apply_smoothing_in_emission_matrix = False
        self.apply_smoothing_in_transition_matrix = False

    def _set_attributes(
        self,
        data: List[Tuple[str, str]],
        use_log_prob: bool,
        smoothing_factor: Optional[float],
        apply_smoothing_in_emission_matrix: bool,
        apply_smoothing_in_transition_matrix: bool,
    ):
        """
        Set the attributes of the HMM based on the given data.

        Input:
            data:
                usually is the train data
            use_log_prob & smoothing_factor:
                both is used for calculating the probability
            apply_smoothing_in_emission_matrix & apply_smoothing_in_transition_matrix:
                both is used for smoothing, if set True then the smoothing will be applied
                when calculating the probability in the matrix

            (for further description can be read at the train func)
        """
        words, tags = set(), set()
        for word, tag in data:
            words.add(word)
            tags.add(tag)

        if UNK_TOKEN not in tags:
            tags.add(UNK_TOKEN)
        if START_TOKEN not in tags:
            tags.add(START_TOKEN)
        if STOP_TOKEN not in tags:
            tags.add(STOP_TOKEN)
        self.Q = list(tags)
        self.O = list(words)
        self.N = len(self.Q)
        self.T = len(self.O)

        self.use_log_prob = use_log_prob
        self.smoothing_factor = (
            smoothing_factor if smoothing_factor else self.smoothing_factor
        )
        self.phi = [self.smoothing_factor] * self.N

        self.apply_smoothing_in_emission_matrix = apply_smoothing_in_emission_matrix
        self.apply_smoothing_in_transition_matrix = apply_smoothing_in_transition_matrix

    def train(
        self,
        train_data: List[Tuple[str, str]],
        use_log_prob: bool = False,
        smoothing_factor: Optional[float] = None,
        apply_smoothing_in_emission_matrix: bool = False,
        apply_smoothing_in_transition_matrix: bool = False,
        have_start_stop_tokens_exists: bool = False,
    ):
        """
        Train the HMM model based on the given data.

        Input:
            train_data:
                list of pairs (word & tag)
                eg.
                [('The', 'DET'), ('Fulton', 'NOUN'), ...]

                meaning:
                - train_data[0] = 1st pair of word & tag

            use_log_prob:
                A bool to determine whether we want the value of probability is logged or not.
                If True, means we need to log the value in matrices.

            smoothing_factor:
                A number that used for smoothing.

            apply_smoothing_in_emission_matrix:
                A bool to determine whether smoothing process will be applied when calculating emission matrix or not

            apply_smoothing_in_transition_matrix:
                A bool to determine whether smoothing process will be applied when calculating transition matrix or not

            have_start_stop_tokens_exists:
                If True means the sentence in train_data already has START_TOKEN at the beginning of the sentence
                    and STOP_TOKEN at the end.
                Else means we need to insert these tokens
        """
        self._init_attributes()
        self._set_attributes(
            train_data,
            use_log_prob,
            smoothing_factor,
            apply_smoothing_in_emission_matrix,
            apply_smoothing_in_transition_matrix,
        )

        if not have_start_stop_tokens_exists:
            self._add_start_and_stop_tokens_in_sentence()

        # Calculating both the Tranisition and Emission Matrix
        self.emission_matrix = self._calculate_emission_matrix(train_data)
        self.transition_matrix = self._calculate_transition_matrix(train_data)

    def evaluate(self, gold_data: List[List[Tuple[str, str]]]):
        """
        Input: list of pairs (word & tag) for each sentence
            read the train function's input for further explanation

        Task:
            - strip the tags from the gold data, retag it using this HMM tagger (use predict function)
            - show / return the evaluation score (eg. accuracy)
        """
        pass

    def predict(self, sent: List[str], need_start_stop_tokens_removed: bool = True):
        """
        Input: List of words in a sentence
        Output: List of tags

        Notes:
        - applying Viterbi algorithm
        - don't forget to handle the UNK token
        """

        if need_start_stop_tokens_removed:
            self._remove_start_and_stop_tokens_in_sentence()
        pass

    def get_transition_matrix(self):
        return self.transition_matrix

    def get_emission_matrix(self):
        return self.emission_matrix

    def _calculate_emission_matrix(self, train_data: List[Tuple[str, str]]):
        """
        Calculates the emission matrix, which represents the conditional probabilities of words given tags.

        Input:
            train_data:
                A list of pairs (word & tag).

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

        Example Calculation of the Probability in Emission Matrix:
            P(oi|qi) = count(VERB, 'be') / count(VERB) = 4046/13126 = 0.31
        """

        tag_word_counts = {}  # Dictionary to store word counts for each tag
        tag_counts = {}  # Dictionary to store tag counts
        emission_matrix = {}  # Dictionary to store the emission matrix

        # Aliasing self.Q as tag_list for readable-purpose
        # and remove START_TOKEN & STOP_TOKEN in tag_list because emission_matrix don't need it
        tag_list = list(self.Q)
        tag_list.remove(START_TOKEN)
        tag_list.remove(STOP_TOKEN)

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
                temp_matrix[word] = self._get_probability(
                    word_count,
                    tag_counts[tag],
                    self.use_log_prob,
                    self.apply_smoothing_in_emission_matrix,
                    self.smoothing_factor,
                )

            # Store the tag's emission probabilities in the emission matrix
            emission_matrix[tag] = temp_matrix

        return emission_matrix

    def _calculate_transition_matrix(
        self,
        train_data: List[Tuple[str, str]],
    ):
        """
        Calculates the transition matrix, which represents the probabilities of a specific tag given a previous tag.

        Formula: Transition_Probability(Tag2 | Tag1)

        Input:
            train_data:
                A list of pairs (word & tag).

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

        Example Calculation of the Probability in Transition Matrix:
            P(qi|qi−1) = count(AUX,VERB) / count(AUX) = 10471/13124 = .80
        """
        # Initialize an empty dictionary to store the transition matrix
        transition_matrix = {}

        # Aliasing self.Q as tag_list for readable-purpose
        tag_list = self.Q

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
                transition_matrix[t2][t1] = self._get_probability(
                    count_t2_t1,
                    tag_counts[t1],
                    self.use_log_prob,
                    self.apply_smoothing_in_transition_matrix,
                    self.smoothing_factor,
                )

        # Return the computed transition matrix
        return transition_matrix

    def _get_probability(
        self,
        numerator: float,
        denominator: float,
        use_log_prob: bool,
        is_smoothing_applied: bool,
        smoothing_factor: float,
    ):
        """
        Calculate the probability used in emission matrix / transition matrix.
        There are options to calculate the probabilty:
        - use log / not (param: use_log_prob)
        - do smoothing / not (param: is_smoothing_applied)
        -  how much the smoothing factor is used (param: smoothing_factor; only worked when is_smoothing_applied = True)

        Input:
            numerator: float
            denominator: float
            use_log_prob: bool
            is_smoothing_applied: bool
            smoothing_factor: float

        Output:
            value:
                - Type: float
                - Example: 0.001, 2.0, -inf
        """
        if not use_log_prob:
            if not is_smoothing_applied:
                value = numerator / denominator
            else:
                if not smoothing_factor:
                    print(
                        "WARNING: Smoothing is applied but smoothing_factor is set to 0!"
                    )
                value = (numerator + smoothing_factor) / (
                    denominator + smoothing_factor
                )
        else:
            # rules of log: log(A/B) - log A - log B
            if not is_smoothing_applied:
                # if no smoothing factor, means numerator or denominator maybe 0
                if not numerator or not denominator:
                    # case: numerator = 0 & denominator = else, result is -inf. math lib can't operate that
                    # case: numerator = 0 & denominator = 0, result NaN. math lib can't operate that
                    value = -math.inf
                else:
                    value = math.log(numerator) - math.log(denominator)
            else:
                if not smoothing_factor:
                    print(
                        "WARNING: Smoothing is applied but smoothing_factor is set to 0!"
                    )
                value = math.log(numerator + smoothing_factor) - math.log(
                    denominator + smoothing_factor
                )

        return value

    def _add_start_and_stop_tokens_in_sentence():
        pass

    def _remove_start_and_stop_tokens_in_sentence():
        pass
