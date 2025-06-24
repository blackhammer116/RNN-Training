# **RNN Training Implementation**
This project Implements 3 of RNN's variants with their own use-cases and datasets. These include Vannila RNN, LSTM (Long-short-Term-Memory) RNN and GRU.

## Table of content
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [RNN Variants](#rnn-variants)
    - [Vanilla RNN](#vanilla-rnn)
    - [LSTM](#lstm)
    - [GRU](#gru)
- [Datasets](#datasets)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction
Recurrent Neural Networks (RNNs) are a class of neural networks designed for sequence prediction problems. This project demonstrates the implementation of three RNN variants: Vanilla RNN, LSTM, and GRU, showcasing their unique strengths and applications.

## Installation
To set up the project, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/your-repo/RNN-Training.git
    cd RNN-Training
    ```
2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
Run the training script with the desired RNN variant and dataset:
```bash
python main.py <model_type>
```
Replace `<model_type>` with `vrnn`, `lstm`, or `gru`

## RNN Variants

### Vanilla RNN
The Vanilla RNN is the simplest form of RNN, suitable for basic sequence modeling tasks. However, it suffers from vanishing gradient issues for long sequences. Used for simple charater level prediction or when the sequence is not too large

### LSTM
LSTM networks address the vanishing gradient problem by introducing memory cells and gates, making them effective for long-term dependencies. Used often for sentiment analysis task

### GRU
GRU is a simplified version of LSTM with fewer parameters, offering a balance between performance and computational efficiency. used often for time-series/dependent task

## Datasets
The project includes the following datasets:
- **Names Dataset**: For simple character level prediction.
- **Time Series Dataset**: For forecasting and trend analysis.
- **IMDB Movie Review**: For sentiment analysis task.

## Results
The results of the training process, including accuracy and loss metrics, are returned to the console after execution is finished.

