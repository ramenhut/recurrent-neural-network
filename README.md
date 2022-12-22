# Recurrent Neural Network
A simple implementation of a recurrent neural network designed to learn the writing style of an author. Borrows heavily from textbooks and other sources, with support for character and word based tokenization. This is far from the best approach (network design, stack, etc.) for this particular task. Run the script using the following command, which will attempt to load a training set from text files in a \shakes subdirectory, train the network, and periodically output sample text:

    python rnn.py

When training with the complete works of Shakespeare, this network produces results such as the following, after a few epochs:

    Two are pardoned a life, that every clap is his church. 
    Who, wherefore are I to be endured for thus depart, your Roman suns. 

    You be strange haste, that I royal caterpillars coward.
    'Tis make which gold, being unto the counterpoise, harmless

    We'll nuptial once, to suck disobedience
    And clapp'd positively sometime comes. But fell not no thing must be
    but your consequence; monstrous John?

    To given I live in array, hearts,
    And not in the arms than I that hath March

    I could but hear you encounter it,
    In our brothers Oxford now but Edward, say'st?

    Speak thou go.
    They I come

More information available at [bertolami.com](https://bertolami.com).