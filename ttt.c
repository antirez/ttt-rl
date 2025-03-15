#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <float.h>
#include <string.h>
#include <math.h>

// Neural network parameters.
#define NN_INPUT_SIZE 18
#define NN_HIDDEN_SIZE 100
#define NN_OUTPUT_SIZE 9
#define LEARNING_RATE 0.1
#define WINNING_STATES 8

static const int WINNING_PATTERNS[WINNING_STATES][3] = {
    {0, 1, 2},{3, 4, 5},{6, 7, 8},  // Horizontal
    {0, 3, 6},{1, 4, 7},{2, 5, 8},  // Vertical
    {0, 4, 8},{2, 4, 6}             // Diagonal
};

// Game board representation.
typedef struct {
    char board[9];          // Can be "." (empty) or "X", "O".
    int current_player;     // 0 for player (X), 1 for computer (O).
} GameState;

/* Neural network structure. For simplicity we have just
 * one hidden layer and fixed sizes (see defines above).
 * However for this problem going deeper than one hidden layer
 * is useless. */
typedef struct {
    // Weights and biases.
    float weights_ih[NN_INPUT_SIZE * NN_HIDDEN_SIZE];
    float weights_ho[NN_HIDDEN_SIZE * NN_OUTPUT_SIZE];
    float biases_h[NN_HIDDEN_SIZE];
    float biases_o[NN_OUTPUT_SIZE];

    // Activations are part of the structure itself for simplicity.
    float inputs[NN_INPUT_SIZE];
    float hidden[NN_HIDDEN_SIZE];
    float raw_logits[NN_OUTPUT_SIZE]; // Outputs before softmax().
    float outputs[NN_OUTPUT_SIZE];    // Outputs after softmax().
} NeuralNetwork;

/* ReLU activation function */
float relu(float x) {
    return x > 0 ? x : 0;
}

/* Derivative of ReLU activation function */
float relu_derivative(float x) {
    return x > 0 ? 1.0f : 0.0f;
}

/* Initialize a neural network with random weights, we should
 * use something like He weights since we use RELU, but we don't
 * care as this is a trivial example. */
#define RANDOM_WEIGHT() (((float)rand() / RAND_MAX) - 0.5f)
void init_neural_network(NeuralNetwork *nn) {
    // Initialize weights with random values between -0.5 and 0.5
    for (int i = 0; i < NN_INPUT_SIZE * NN_HIDDEN_SIZE; i++)
        nn->weights_ih[i] = RANDOM_WEIGHT();

    for (int i = 0; i < NN_HIDDEN_SIZE * NN_OUTPUT_SIZE; i++)
        nn->weights_ho[i] = RANDOM_WEIGHT();

    for (int i = 0; i < NN_HIDDEN_SIZE; i++)
        nn->biases_h[i] = RANDOM_WEIGHT();

    for (int i = 0; i < NN_OUTPUT_SIZE; i++)
        nn->biases_o[i] = RANDOM_WEIGHT();
}

/* Apply softmax activation function to an array input, and
 * set the result into output. */
void softmax(float *input, float *output, int size) {
    /* Find maximum value then subtact it to avoid
     * numerical stability issues with exp(). */
    float max_val = input[0];
    for (int i = 1; i < size; i++) {
        if (input[i] > max_val) {
            max_val = input[i];
        }
    }

    // Calculate exp(x_i - max) for each element and sum.
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        output[i] = expf(input[i] - max_val);
        sum += output[i];
    }

    // Normalize to get probabilities.
    if (sum > 0) {
        for (int i = 0; i < size; i++) {
            output[i] /= sum;
        }
    } else {
        /* Fallback in case of numerical issues, just provide
         * a uniform distribution. */
        for (int i = 0; i < size; i++) {
            output[i] = 1.0f / size;
        }
    }
}

/* Neural network foward pass (inference). We store the activations
 * so we can also do backpropagation later. */
void forward_pass(NeuralNetwork *nn, float *inputs) {
    // Copy inputs.
    memcpy(nn->inputs, inputs, NN_INPUT_SIZE * sizeof(float));

    // Input to hidden layer.
    for (int i = 0; i < NN_HIDDEN_SIZE; i++) {
        float sum = nn->biases_h[i];
        for (int j = 0; j < NN_INPUT_SIZE; j++) {
            sum += inputs[j] * nn->weights_ih[j * NN_HIDDEN_SIZE + i];
        }
        nn->hidden[i] = relu(sum);
    }

    // Hidden to output (raw logits).
    for (int i = 0; i < NN_OUTPUT_SIZE; i++) {
        nn->raw_logits[i] = nn->biases_o[i];
        for (int j = 0; j < NN_HIDDEN_SIZE; j++) {
            nn->raw_logits[i] += nn->hidden[j] * nn->weights_ho[j * NN_OUTPUT_SIZE + i];
        }
    }

    // Apply softmax to get the final probabilities.
    softmax(nn->raw_logits, nn->outputs, NN_OUTPUT_SIZE);
}

/* Initialize game state with an empty board. */
void init_game(GameState *state) {
    memset(state->board,'.',9);
    state->current_player = 0;  // Player (X) goes first
}

/* Show board on screen in ASCII "art"... */
void display_board(GameState *state) {
    for (int row = 0; row < 3; row++) {
        // Display the board symbols.
        printf("%c%c%c ", state->board[row*3], state->board[row*3+1],
                          state->board[row*3+2]);

        // Display the position numbers for this row, for the poor human.
        printf("%d%d%d\n", row*3, row*3+1, row*3+2);
    }
    printf("\n");
}

/* Convert board state to neural network inputs. Note that we use
 * a peculiar encoding I descrived here:
 * https://www.youtube.com/watch?v=EXbgUXt8fFU
 *
 * Instead of one-hot encoding, we can represent N different categories
 * as different bit patterns. In this specific case it's trivial:
 *
 * 00 = empty
 * 10 = X
 * 01 = O
 *
 * Two inputs per symbol instead of 3 in this case, but in the general case
 * this reduces the input dimensionality A LOT.
 *
 * LEARNING OPPORTUNITY: You may want to learn (if not already aware) of
 * different ways to represent non scalar inputs in neural networks:
 * One hot encoding, learned embeddings, and even if it's just my random
 * exeriment this "permutation coding" that I'm using here.
 */
void board_to_inputs(GameState *state, float *inputs) {
    for (int i = 0; i < 9; i++) {
        if (state->board[i] == '.') {
            inputs[i*2] = 0;
            inputs[i*2+1] = 0;
        } else if (state->board[i] == 'X') {
            inputs[i*2] = 1;
            inputs[i*2+1] = 0;
        } else {  // 'O'
            inputs[i*2] = 0;
            inputs[i*2+1] = 1;
        }
    }
}

/* Check if the game is over (win or tie).
 * Very brutal but fast enough. */
int check_game_over(GameState *state, char *winner) {
    // Check if game board match a winning state
    for (int i = 0; i < WINNING_STATES; i++) {
        char player = state->board[WINNING_PATTERNS[i][0]];
        
        if (player != '.' &&
            player == state->board[WINNING_PATTERNS[i][1]] &&
            player == state->board[WINNING_PATTERNS[i][2]]) {
            *winner = player;
            return 1;
        }
    }

    // Check for tie (no free tiles left).
    for(int i = 0; i < 9; i++) {
        if(state->board[i] == '.') {
            break;
        }

        // Last iteration hence tie
        if(i == 8) {
            *winner = 'T';
            return 1;
        }
    }

    return 0; // Game continues.
}

/* Get the best move for the computer using the neural network.
 * Note that there is no complex sampling at all, we just get
 * the output with the highest value THAT has an empty tile. */
int get_computer_move(GameState *state, NeuralNetwork *nn, int display_probs) {
    float inputs[NN_INPUT_SIZE];

    board_to_inputs(state, inputs);
    forward_pass(nn, inputs);

    // Find the highest probability value and best legal move.
    float highest_prob = -1.0f;
    int highest_prob_idx = -1;
    int best_move = -1;
    float best_legal_prob = -1.0f;

    for (int i = 0; i < 9; i++) {
        // Track highest probability overall.
        if (nn->outputs[i] > highest_prob) {
            highest_prob = nn->outputs[i];
            highest_prob_idx = i;
        }

        // Track best legal move.
        if (state->board[i] == '.' &&
            (best_move == -1 || nn->outputs[i] > best_legal_prob))
        {
            best_move = i;
            best_legal_prob = nn->outputs[i];
        }
    }

    // That's just for debugging. It's interesting to show to user
    // in the first iterations of the game, since you can see how initially
    // the net picks illegal moves as best, and so forth.
    if (display_probs) {
        printf("Neural network move probabilities:\n");
        for (int row = 0; row < 3; row++) {
            for (int col = 0; col < 3; col++) {
                int pos = row * 3 + col;

                // Print probability as percentage.
                printf("%5.1f%%", nn->outputs[pos] * 100.0f);

                // Add markers.
                if (pos == highest_prob_idx) {
                    printf("*"); // Highest probability overall.
                }
                if (pos == best_move) {
                    printf("#"); // Selected move (highest valid probability).
                }
                printf(" ");
            }
            printf("\n");
        }

        // Sum of probabilities should be 1.0, hopefully.
        // Just debugging.
        float total_prob = 0.0f;
        for (int i = 0; i < 9; i++)
            total_prob += nn->outputs[i];
        printf("Sum of all probabilities: %.2f\n\n", total_prob);
    }
    return best_move;
}

/* Backpropagation function.
 * The only difference here from vanilla backprop is that we have
 * a 'reward_scaling' argument that makes the output error more/less
 * dramatic, so that we can adjust the weights proportionally to the
 * reward we want to provide. */
void backprop(NeuralNetwork *nn, float *target_probs, float learning_rate, float reward_scaling) {
    float output_deltas[NN_OUTPUT_SIZE];
    float hidden_deltas[NN_HIDDEN_SIZE];

    /* === STEP 1: Compute deltas === */

    /* Calculate output layer deltas:
     * Note what's going on here: we are technically using softmax
     * as output function and cross entropy as loss, but we never use
     * cross entropy in practice since we check the progresses in terms
     * of winning the game.
     *
     * Still calculating the deltas in the output as:
     *
     *      output[i] - target[i]
     *
     * Is exactly what happens if you derivate the deltas with
     * softmax and cross entropy.
     *
     * LEARNING OPPORTUNITY: This is a well established and fundamental
     * result in neural networks, you may want to read more about it. */
    for (int i = 0; i < NN_OUTPUT_SIZE; i++) {
        output_deltas[i] =
            (nn->outputs[i] - target_probs[i]) * fabsf(reward_scaling);
    }

    // Backpropagate error to hidden layer.
    for (int i = 0; i < NN_HIDDEN_SIZE; i++) {
        float error = 0;
        for (int j = 0; j < NN_OUTPUT_SIZE; j++) {
            error += output_deltas[j] * nn->weights_ho[i * NN_OUTPUT_SIZE + j];
        }
        hidden_deltas[i] = error * relu_derivative(nn->hidden[i]);
    }

    /* === STEP 2: Weights updating === */

    // Output layer weights and biases.
    for (int i = 0; i < NN_HIDDEN_SIZE; i++) {
        for (int j = 0; j < NN_OUTPUT_SIZE; j++) {
            nn->weights_ho[i * NN_OUTPUT_SIZE + j] -=
                learning_rate * output_deltas[j] * nn->hidden[i];
        }
    }
    for (int j = 0; j < NN_OUTPUT_SIZE; j++) {
        nn->biases_o[j] -= learning_rate * output_deltas[j];
    }

    // Hidden layer weights and biases.
    for (int i = 0; i < NN_INPUT_SIZE; i++) {
        for (int j = 0; j < NN_HIDDEN_SIZE; j++) {
            nn->weights_ih[i * NN_HIDDEN_SIZE + j] -=
                learning_rate * hidden_deltas[j] * nn->inputs[i];
        }
    }
    for (int j = 0; j < NN_HIDDEN_SIZE; j++) {
        nn->biases_h[j] -= learning_rate * hidden_deltas[j];
    }
}

/* Train the neural network based on game outcome.
 *
 * The move_history is just an integer array with the index of all the
 * moves. This function is designed so that you can specify if the
 * game was started by the move by the NN or human, but actually the
 * code always let the human move first. */
void learn_from_game(NeuralNetwork *nn, int *move_history, int num_moves, int nn_moves_even, char winner) {
    // Determine reward based on game outcome
    float reward;
    char nn_symbol = nn_moves_even ? 'O' : 'X';

    if (winner == 'T') {
        reward = 0.3f;  // Small reward for draw
    } else if (winner == nn_symbol) {
        reward = 1.0f;  // Large reward for win
    } else {
        reward = -2.0f; // Negative reward for loss
    }

    GameState state;
    float target_probs[NN_OUTPUT_SIZE];

    // Process each move the neural network made.
    for (int move_idx = 0; move_idx < num_moves; move_idx++) {
        // Skip if this wasn't a move by the neural network.
        if ((nn_moves_even && move_idx % 2 != 1) ||
            (!nn_moves_even && move_idx % 2 != 0))
        {
            continue;
        }

        // Recreate board state BEFORE this move was made.
        init_game(&state);
        for (int i = 0; i < move_idx; i++) {
            char symbol = (i % 2 == 0) ? 'X' : 'O';
            state.board[move_history[i]] = symbol;
        }

        // Convert board to inputs and do forward pass.
        float inputs[NN_INPUT_SIZE];
        board_to_inputs(&state, inputs);
        forward_pass(nn, inputs);

        /* The move that was actually made by the NN, that is
         * the one we want to reward (positively or negatively). */
        int move = move_history[move_idx];

        /* Here we can't really implement temporal difference in the strict
         * reinforcement learning sense, since we don't have an easy way to
         * evaluate if the current situation is better or worse than the
         * previous state in the game.
         *
         * However "time related" we do something that is very effective in
         * this case: we scale the reward according to the move time, so that
         * later moves are more impacted (the game is less open to different
         * solutions as we go forward).
         *
         * We give a fixed 0.5 importance to all the moves plus
         * a 0.5 that depends on the move position.
         *
         * NOTE: this makes A LOT of difference. Experiment with different
         * values.
         *
         * LEARNING OPPORTUNITY: Temporal Difference in Reinforcement Learning
         * is a very important result, that was worth the Turing Award in
         * 2024 to Sutton and Barto. You may want to read about it. */
        float move_importance = 0.5f + 0.5f * (float)move_idx/(float)num_moves;
        float scaled_reward = reward * move_importance;

        /* Create target probability distribution:
         * let's start with the logits all set to 0. */
        for (int i = 0; i < NN_OUTPUT_SIZE; i++)
            target_probs[i] = 0;

        /* Set the target for the chosen move based on reward: */
        if (scaled_reward >= 0) {
            /* For positive reward, set probability of the chosen move to
             * 1, with all the rest set to 0. */
            target_probs[move] = 1;
        } else {
            /* For negative reward, distribute probability to OTHER
             * valid moves, which is conceptually the same as discouraging
             * the move that we want to discourage. */
            int valid_moves_left = 9-move_idx-1;
            float other_prob = 1.0f / valid_moves_left;
            for (int i = 0; i < 9; i++) {
                if (state.board[i] == '.' && i != move) {
                    target_probs[i] = other_prob;
                }
            }
        }

        /* Call the generic backpropagation function, using
         * our target logits as target. */
        backprop(nn, target_probs, LEARNING_RATE, scaled_reward);
    }
}

/* Play one game of Tic Tac Toe against the neural network. */
void play_game(NeuralNetwork *nn) {
    GameState state;
    char winner;
    int move_history[9]; // Maximum 9 moves in a game.
    int num_moves = 0;

    init_game(&state);

    printf("Welcome to Tic Tac Toe! You are X, the computer is O.\n");
    printf("Enter positions as numbers from 0 to 8 (see picture).\n");

    while (!check_game_over(&state, &winner)) {
        display_board(&state);

        if (state.current_player == 0) {
            // Human turn.
            int move;
            char movec;
            printf("Your move (0-8): ");
            scanf(" %c", &movec);
            move = movec-'0'; // Turn character into number.

            // Check if move is valid.
            if (move < 0 || move > 8 || state.board[move] != '.') {
                printf("Invalid move! Try again.\n");
                continue;
            }

            state.board[move] = 'X';
            move_history[num_moves++] = move;
        } else {
            // Computer's turn
            printf("Computer's move:\n");
            int move = get_computer_move(&state, nn, 1);
            state.board[move] = 'O';
            printf("Computer placed O at position %d\n", move);
            move_history[num_moves++] = move;
        }

        state.current_player = !state.current_player;
    }

    display_board(&state);

    if (winner == 'X') {
        printf("You win!\n");
    } else if (winner == 'O') {
        printf("Computer wins!\n");
    } else {
        printf("It's a tie!\n");
    }

    // Learn from this game
    learn_from_game(nn, move_history, num_moves, 1, winner);
}

/* Get a random valid move, this is used for training
 * against a random opponent. Note: this function will loop forever
 * if the board is full, but here we want simple code. */
int get_random_move(GameState *state) {
    while(1) {
        int move = rand() % 9;
        if (state->board[move] != '.') continue;
        return move;
    }
}

/* Play a game against random moves and learn from it.
 *
 * This is a very simple Montecarlo Method applied to reinforcement
 * learning:
 *
 * 1. We play a complete random game (episode).
 * 2. We determine the reward based on the outcome of the game.
 * 3. We update the neural network in order to maximize future rewards.
 *
 * LEARNING OPPORTUNITY: while the code uses some Montecarlo-alike
 * technique, important results were recently obtained using
 * Montecarlo Tree Search (MCTS), where a tree structure repesents
 * potential future game states that are explored according to
 * some selection: you may want to learn about it. */
char play_random_game(NeuralNetwork *nn, int *move_history, int *num_moves) {
    GameState state;
    char winner = 0;
    *num_moves = 0;

    init_game(&state);

    while (!check_game_over(&state, &winner)) {
        int move;

        if (state.current_player == 0) {  // Random player's turn (X)
            move = get_random_move(&state);
        } else {  // Neural network's turn (O)
            move = get_computer_move(&state, nn, 0);
        }

        /* Make the move and store it: we need the moves sequence
         * during the learning stage. */
        char symbol = (state.current_player == 0) ? 'X' : 'O';
        state.board[move] = symbol;
        move_history[(*num_moves)++] = move;

        // Switch player.
        state.current_player = !state.current_player;
    }

    // Learn from this game - neural network is 'O' (even-numbered moves).
    learn_from_game(nn, move_history, *num_moves, 1, winner);
    return winner;
}

/* Train the neural network against random moves. */
void train_against_random(NeuralNetwork *nn, int num_games) {
    int move_history[9];
    int num_moves;
    int wins = 0, losses = 0, ties = 0;

    printf("Training neural network against %d random games...\n", num_games);

    int played_games = 0;
    for (int i = 0; i < num_games; i++) {
        char winner = play_random_game(nn, move_history, &num_moves);
        played_games++;

        // Accumulate statistics that are provided to the user (it's fun).
        if (winner == 'O') {
            wins++; // Neural network won.
        } else if (winner == 'X') {
            losses++; // Random player won.
        } else {
            ties++; // Tie.
        }

        // Show progress every many games to avoid flooding the stdout.
        if ((i + 1) % 10000 == 0) {
            printf("Games: %d, Wins: %d (%.1f%%), "
                   "Losses: %d (%.1f%%), Ties: %d (%.1f%%)\n",
                  i + 1, wins, (float)wins * 100 / played_games,
                  losses, (float)losses * 100 / played_games,
                  ties, (float)ties * 100 / played_games);
            played_games = 0;
            wins = 0;
            losses = 0;
            ties = 0;
        }
    }
    printf("\nTraining complete!\n");
}

int main(int argc, char **argv) {
    int random_games = 150000; // Fast and enough to play in a decent way.

    if (argc > 1) random_games = atoi(argv[1]);
    srand(time(NULL));

    // Initialize neural network.
    NeuralNetwork nn;
    init_neural_network(&nn);

    // Train against random moves.
    if (random_games > 0) train_against_random(&nn, random_games);

    // Play game with human and learn more.
    while(1) {
        char play_again;
        play_game(&nn);

        printf("Play again? (y/n): ");
        scanf(" %c", &play_again);
        if (play_again != 'y' && play_again != 'Y') break;
    }
    return 0;
}
