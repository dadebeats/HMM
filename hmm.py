from typing import List, Dict, Tuple, Optional
import math


UNK_TOKEN = "UNK"
START_TOKEN = "START"
STOP_TOKEN = "STOP"


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
    - transition_matrix:
        represent A (transition probability matrix). use type dictionary
    - emission_matrix:
        represent B (sequence of observation likelihoods). use type dictionary
    - phi:
        use type dictionary
    - use_log_prob:
        when True, then the value in emission / transition matrix is calculated using log.
        why? probability is a decimal number, when it is not a log value and used in multiplications, it may lead to underflow
    - apply_smoothing_in_emission_matrix:
        when True, then smoothing will be applied in the emission matrix.
        smoothing used to give unseen events a small amount of probability.
    - apply_smoothing_in_transition_matrix:
        when True, then smoothing will be applied in the transition matrix.
        smoothing used to give unseen events a small amount of probability.
    - smoothing_factor:
        is the number for smoothing process

    Notes:
        in this HMM class, we use additional tokens (START_TOKEN, STOP_TOKEN, UNK_TOKEN)
    """

    def __init__(self):
        self._init_attributes()

    def _init_attributes(self):
        self.N = 0  # total tags
        self.T = 0  # total words
        self.Q: List = []  # list tags size N
        self.O: List = []  # list words size T
        self.transition_matrix = {}  # transition_matrix (or A)
        self.emission_matrix = {}  # emission_matrix (or B)
        self.phi = {}  # initial probability distribution
        self.INIT_PROB_VALUE = -math.inf

        # other attributes for tuning the models
        self.use_log_prob = False
        self.apply_smoothing_in_emission_matrix = False
        self.apply_smoothing_in_transition_matrix = False
        self.smoothing_factor = 0.0  # default value

    def _set_attributes(
        self,
        data: List[List[Tuple[str, str]]],
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
        for sent in data:
            for word, tag in sent:
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

        self.apply_smoothing_in_emission_matrix = apply_smoothing_in_emission_matrix
        self.apply_smoothing_in_transition_matrix = apply_smoothing_in_transition_matrix

    def train(
        self,
        train_data: List[List[Tuple[str, str]]],
        use_log_prob: bool = False,
        smoothing_factor: Optional[float] = None,
        apply_smoothing_in_emission_matrix: bool = False,
        apply_smoothing_in_transition_matrix: bool = False,
        ):
        """
        Train the HMM model based on the given data.

        Input:
            train_data:
                A list of pairs (word & tag) for each sentence.
                eg.
                [[('The', 'DET'), ('Fulton', 'NOUN'), ...],
                [],
                []]

                meaning:
                - train_data[0] = List of pairs in the 1st sentence
                - train_data[0][0] = 1st pair of word & tag of the 1st sentence

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

        train_data = [
            self._add_start_and_stop_tokens_in_sentence(p) for p in train_data
        ]

        # Calculating the phi
        self.phi = self._calculate_phi(train_data)
        
        # Calculating both the Tranisition and Emission Matrix
        self.emission_matrix = self._calculate_emission_matrix(train_data)
        self.transition_matrix = self._calculate_transition_matrix(train_data)

    def predict(self, data: List[List[str]]):
        """
        Predict the tags in the sentences using Viterbi Algorithm

        Input:
            data:
            A list of sentences which contains list of words.
            eg.
                [
                    ['I', 'want', 'apple'],
                    ['Tony', 'went', 'to', ...],
                ]

            need_start_stop_tokens_removed:
                If True means the remove the START_TOKEN & STOP_TOKEN from the output tags.
        Output: List of tags of each sentence
        """
        predictions = []
        for sent in data:
            sent = [word.lower() for word in sent]
            tokens = self._viterbi_algorithm(sent)
            predictions.append(tokens)

        # predictions = [
        #     self._remove_start_and_stop_tokens_in_prediction(p) for p in predictions
        # ]

        return predictions
    
    def accuracy(self, data: List[List[Tuple[str, str]]]):
        pass

    def _viterbi_algorithm(self, sent: List[str]):
        """
        Initialization Step:
            viterbi[q, 1]   = πq ∗ bq(o1)
                            = P(token) * P(word|token)
        Recursive Steps:
            viterbi[q,t]    = max viterbi[q',t − 1] ∗ A[q',q] ∗ bq(ot)
                            = max(prev viterbi) * P(token|prev token) * P(word|token)

        Notes:
            - P(token) is from phi
            - P(word|token) is from Emission Matrix
            - P(token|prev token) is from Transition Matrix
        """

        def is_unk_tag(word: str):
            # if the word is not found in the emission_matrix, means the word is not showing up in the train_data
            # hence, it is an unknown word - is_unk = True
            is_unk = True
            for tag in self.Q:
                if word in self.emission_matrix[tag]:
                    is_unk = False
                    break
            return is_unk

        def get_best_tokens(probabilities: Dict[str, float]):
            """
            The best probability (the maximum value) means the token/tag is the best path/option
            
            Output:
                best_tokens (multiple tokens): List[str]
                    Multiple tokens may be generated as the probability values for each token can be identical to one another.
                    eg. When calculating using non-log, most of the probability will likely be 0.
            """
            best_tokens, best_prob = [], self.INIT_PROB_VALUE
            for token, prob in probabilities.items():
                if prob == math.inf or prob == -math.inf:
                    # example case of math.inf:
                    # when calculating using non-log, and emission (-inf) * transition (-inf) resulting inf
                    continue
                if prob > best_prob:
                    best_tokens = [token]
                    best_prob = prob
                elif prob == best_prob:
                    best_tokens.append(token)

            return best_tokens, best_prob

        bestpath_tags = []
        max_prev_viterbi = 0
        backpointer_tag = None
        current_tag = None
        for idx, word in enumerate(sent):
            if is_unk_tag(word):
                # continue to next word
                current_tag = UNK_TOKEN
                bestpath_tags.append(current_tag)
                backpointer_tag = current_tag
                max_prev_viterbi = 0
                continue

            path_probabilities = {}
            if idx == 0:
                # Initialization Step
                for tag in self.Q:
                    # P(tag|token_start) * emission P(word|token)
                    if word not in self.emission_matrix[tag]:
                        prob = self.INIT_PROB_VALUE
                    else:
                        if self.use_log_prob:
                            prob = self.phi[tag] + self.emission_matrix[tag][word]
                        else:
                            prob = self.phi[tag] * self.emission_matrix[tag][word]
                    path_probabilities[tag] = prob

                backpointer_tag = START_TOKEN

            else:
                # Recursive Steps
                for tag in self.Q:
                    # max(prev viterbi) * P(token|prev token) * P(word|token)
                    if backpointer_tag == UNK_TOKEN:
                        transition_value = 0
                    else:
                        # when there is a tag given backpointer_tag that never shown up in the train_data
                        # it is set to -inf
                        transition_value = (
                            self.INIT_PROB_VALUE
                            if backpointer_tag not in self.transition_matrix[tag]
                            else self.transition_matrix[tag][backpointer_tag]
                        )
                    if word not in self.emission_matrix[tag]:
                        # the word is seen in the train_data, but we never got the word with given tag
                        # hence, it is set to -inf
                        emission_value = self.INIT_PROB_VALUE
                    else:
                        emission_value = (
                            self.emission_matrix[tag][word] if tag != STOP_TOKEN else 0
                        )

                    if self.use_log_prob:
                        prob = max_prev_viterbi + transition_value + emission_value
                    else:
                        prob = max_prev_viterbi * transition_value * emission_value
                    path_probabilities[tag] = prob

            best_tokens, best_prob = get_best_tokens(path_probabilities)
            if len(best_tokens) > 1:
                print(
                    "WARNING: Multiple possibilities of tags found for word '" + word + "' with probabilites of " + str(best_prob) + ": " + str(best_tokens)
                )
                print("-- automatically select first possible tag: " + best_tokens[0])
            best_token = best_tokens[0]
            if not best_token:
                # all of the calculation resulting -inf, meaning there is no best option
                best_token = UNK_TOKEN

            current_tag = best_token
            bestpath_tags.append(current_tag)
            backpointer_tag = current_tag
            max_prev_viterbi = 0 if best_prob == self.INIT_PROB_VALUE else best_prob

        return bestpath_tags

    def _calculate_phi(self, train_data: List[List[Tuple[str, str]]]):
        """
        Calculate the phi (probability of each token in initial state) - P(token | START_TOKEN)

        Formula:
            P(token | START_TOKEN)

        Example:
            phi-token = P(token| START_TOKEN) = count(START_TOKEN, token) / count(START_TOKEN)

        Input:
            train_data:
                A list of pairs (word & tag) for each sentence.

        Output:
            phi:
                - Type: Dictionary
                - Structure : {'VERB': 0.001 }
        """
        phi = {}
        total_start_tokens = 0
        tag_start_counts = (
            {}
        )  # save total occurence of a tag with previous token is START_TOKEN
        for sent in train_data:
            if len(sent) > 1:
                try:
                    sent[0] == START_TOKEN
                except Exception as e:
                    raise Exception(
                        "Wrong format, each sentence must initialized with START_TOKEN"
                    ) from e

                token = sent[1]
                total_start_tokens += 1
                if token not in tag_start_counts:
                    tag_start_counts[token] = 0
                tag_start_counts[token] += 1

        denominator = total_start_tokens
        for tag in self.Q:
            if tag == START_TOKEN:
                # it's impossible if START_TOKEN followed by START_TOKEN
                continue
            if tag not in tag_start_counts:
                # tag is not found in training data
                value = self.INIT_PROB_VALUE
            else:
                numerator = tag_start_counts[tag]
                value = self._get_probability(
                    numerator,
                    denominator,
                    self.use_log_prob,
                    self.apply_smoothing_in_transition_matrix,
                    self.smoothing_factor,
                )

            phi[tag] = value

        return phi

    def _calculate_emission_matrix(self, train_data: List[List[Tuple[str, str]]]):
        """
        Calculates the emission matrix, which represents the conditional probabilities of words given tags.

        Example Calculation of the Probability in Emission Matrix:
            P(oi|qi) = count(VERB, 'be') / count(VERB) = 4046/13126 = 0.31

        Input:
            train_data:
                A list sentences consisting of a list of pairs (word & tag).

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

        # Aliasing self.Q as tag_list for readable-purpose
        tag_list = self.Q

        for sentence_data in train_data:
        
            for word, tag in sentence_data:
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

    def _calculate_transition_matrix(self, train_data: List[List[Tuple[str, str]]]):
        """
        Calculates the transition matrix, which represents the probabilities of a specific tag given a previous tag.

        Formula: Transition_Probability(Tag2 | Tag1)

        Example Calculation of the Probability in Transition Matrix:
            P(qi|qi−1) = count(AUX,VERB) / count(AUX) = 10471/13124 = 0.80

        Input:
            train_data:
                A list sentences consisting of pairs (word & tag).

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

        # Aliasing self.Q as tag_list for readable-purpose
        tag_list = self.Q
        # tag_list.remove('UNK')

        # Count occurrences of each tag in the training data
        tag_counts = {'UNK':0}
        for sentence_data in train_data:
            for word, tag in sentence_data:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1

        # Calculate transition probabilities for each pair of tags (t1, t2)
        for t2 in tag_list:
            # Initialize a dictionary to store transition probabilities for t2 given all possible t1 tags
            transition_matrix[t2] = {}
            for t1 in tag_list:
                if t1 == START_TOKEN:
                    # skip because will be handled & saved in the phi attribute
                    continue

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
        value = 0
        if not (numerator + smoothing_factor) or not (denominator + smoothing_factor):
            # if numerator + smoothing_factor = 0
            # or denominator + smoothing_factor = 0
            # automatically set the probability
            value = self.INIT_PROB_VALUE
            return value

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

    def _add_start_and_stop_tokens_in_sentence(self, sent: List[Tuple[str, str]]):
        """
        Input:
            sent:
                A list of pairs (word & tag).

        Output:
            [(START_TOKEN, START_TOKEN)] + sent + [(STOP_TOKEN, STOP_TOKEN)]
            e.g
                [('<start>', 'START')]+sent+[('<stop>','STOP')]
        """
        
        return [(START_TOKEN, START_TOKEN)]+sent+[(STOP_TOKEN, STOP_TOKEN)]

    def _remove_start_and_stop_tokens_in_prediction(self, tags: List[str]):
        """
        Input:
            tags:
                A list of predicted tags.

        Task:
            - if tags[0] == START_TOKEN -> remove
            - if tags[-1] == END_TOKEN -> remove
        """
        pass

    def get_transition_matrix(self):
        return self.transition_matrix
    
    def get_emission_matrix(self):
        return self.emission_matrix