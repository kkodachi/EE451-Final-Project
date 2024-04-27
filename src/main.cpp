#include "../include/module.h"
#include "../include/optim.h"
#include "../include/parser.h"
#include <chrono>

using namespace std;


int main(int argc, char **argv) {
    setbuf(stdout, NULL);
    if (argc < 2) {
        cout << "parallel_gcn graph_name [num_nodes input_dim hidden_dim output_dim"
                "dropout learning_rate, weight_decay epochs early_stopping]" << endl;
        return EXIT_FAILURE;
    }

    // Parse the selected dataset
    GCNParams params = GCNParams::get_default();
    GCNData data;
    std::string input_name(argv[1]);
    Parser parser(&params, &data, input_name);
    if (!parser.parse()) {
        std::cerr << "Cannot read input: " << input_name << std::endl;
        exit(EXIT_FAILURE);
    }
    std::cout << "Starting time here" << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    GCN gcn(params, &data); // Create and initialize and object of type GCN.
    gcn.run(); // Run the main function of the model in order to train and validate the solution.
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << "Total time: " << duration.count() << " ms" << std::endl;
    return EXIT_SUCCESS;
}