# Machine Translation with Attention Mechanism

This repository implements a sequence-to-sequence model with attention for machine translation. The model is trained on an English-to-French translation task, using neural networks to learn the translation between the two languages. The implementation follows a typical architecture of an Encoder-Decoder network, enhanced with Bahdanau attention to improve performance on longer sequences by allowing the decoder to focus on relevant parts of the input sequence.

## Overview

The project includes:

- **Preprocessing**: Tokenization and sentence normalization of input pairs.
- **Encoder**: A GRU-based encoder that converts input sentences into a context vector.
- **Decoder**: A GRU-based decoder that generates output sequences, using Bahdanau attention mechanism to align the input and output sequences.
- **Training**: The model is trained on the English-French translation task using teacher forcing and cross-entropy loss.

The architecture allows the model to learn the translation mapping between English and French sentences, focusing on key words during translation using attention mechanisms.

## Key Features

- **Sequence-to-sequence model**: Uses GRU layers in both the encoder and decoder.
- **Attention mechanism**: Bahdanau attention is implemented to help the model focus on important words in the input sequence during translation.
- **Tokenization and preprocessing**: The dataset is processed by normalizing, removing special characters, and converting sentences to lowercase.
- **Training with teacher forcing**: The model uses teacher forcing during training, where the true target word is provided as the next input to the decoder.
- **Evaluation**: After training, the model can be used to generate translations for new English sentences.

## Architecture

The model consists of the following components:

1. **Encoder**: 
   - Converts input sentences into hidden states using GRU.
   - Embeds input tokens and processes them sequentially.
   
2. **Decoder**: 
   - Uses the hidden state of the encoder to generate translations.
   - Employs the Bahdanau attention mechanism to focus on different parts of the input sequence at each time step.

3. **Attention Mechanism**:
   - The Bahdanau attention mechanism computes attention scores, which are used to generate a context vector.
   - The context vector is then combined with the current hidden state of the decoder to produce the next token.

## Model Training

The model is trained using the following setup:

- **Optimizer**: Adam optimizer is used for updating the parameters of the encoder and decoder.
- **Loss Function**: Negative log-likelihood loss (NLLLoss) is used to calculate the loss between the predicted and target sequences.
- **Batch Size**: The training is done in mini-batches with a batch size of 32.
- **Epochs**: The model is trained for 100 epochs.

## Data Preprocessing

1. **Normalization**: The input and output sentences are normalized by converting them to lowercase and removing special characters (e.g., punctuation and non-alphabetic characters).
2. **Tokenization**: The sentences are split into words and then mapped to integer indices.
3. **Padding**: Sentences are padded to a maximum length of 10 tokens for consistency.

## Training Process

The training loop performs the following:

1. **Forward Pass**: The input sequence is passed through the encoder to get hidden states, and the decoder generates an output sequence.
2. **Loss Calculation**: The output sequence is compared with the target sequence to calculate the loss.
3. **Backpropagation**: The gradients are calculated and the weights are updated using the Adam optimizer.

The model is trained and evaluated at periodic intervals, and the loss is visualized at the end of training.

## Evaluation

Once trained, the model can generate translations for new sentences by passing them through the encoder-decoder network and using the output probabilities to predict the translated sentence. The attention mechanism is visualized by plotting the attention weights.

## Results

After training, the model is capable of translating short English sentences into French. The quality of translations depends on the quality of training data and the length of the sentences.

## Future Improvements

- **Larger Dataset**: The model can be trained on a larger dataset for improved generalization.
- **Bidirectional Encoder**: A bidirectional encoder could be implemented to capture more context from the input sentence.
- **More Complex Attention**: Experimenting with different types of attention mechanisms, such as Luong attention, could further improve performance.

## Contributions

Feel free to fork the repository, make improvements, and submit pull requests.



## Acknowledgments

- The model architecture is inspired by the work of Bahdanau et al. on neural machine translation with attention mechanisms.
- Special thanks to the PyTorch community for providing tools and documentation to implement this model efficiently.

