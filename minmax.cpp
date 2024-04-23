#include <iostream>
#include <vector>
#include <omp.h>
#include <ctime>

using namespace std;

// Kernel function to perform reduction for finding minimum value
int reduce_min(vector<int>& data) {
    int min_val = data[0];
    #pragma omp parallel for reduction(min:min_val)
    for (int i = 0; i < data.size(); ++i) {
        if (data[i] < min_val) {
            min_val = data[i];
        }
    }
    return min_val;
}

// Kernel function to perform reduction for finding maximum value
int reduce_max(vector<int>& data) {
    int max_val = data[0];
    #pragma omp parallel for reduction(max:max_val)
    for (int i = 0; i < data.size(); ++i) {
        if (data[i] > max_val) {
            max_val = data[i];
        }
    }
    return max_val;
}

// Kernel function to perform reduction for finding sum
int reduce_sum(vector<int>& data) {
    int sum = 0;
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < data.size(); ++i) {
        sum += data[i];
    }
    return sum;
}

// Kernel function to perform reduction for finding average
double reduce_avg(vector<int>& data) {
    int sum = reduce_sum(data);
    return static_cast<double>(sum) / data.size();
}

int main() {
    // Generate some random data
    vector<int> data(7);
    srand(time(0));
    for (int i = 0; i < data.size(); ++i) {
        data[i] = rand() % 1000; // Random numbers between 0 and 999
    }

    // Calculate and print min, max, sum, and average using parallel reduction
    for (int i = 0; i < data.size(); ++i) {
        cout << data[i] << " "; // Random numbers between 0 and 999
    }
    cout << endl ;
    
    cout << "Minimum: " << reduce_min(data) << endl;
    cout << "Maximum: " << reduce_max(data) << endl;
    cout << "Sum: " << reduce_sum(data) << endl;
    cout << "Average: " << reduce_avg(data) << endl;

    return 0;
}

// g++ -o graph minmax.cpp -fopenmp
// ./graph