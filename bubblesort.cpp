#include <iostream>
#include <vector>
#include <chrono>
#include <omp.h>

using namespace std;
using namespace std::chrono;

// Sequential Bubble Sort
void bubbleSortSequential(vector<int>& arr) {
    int n = arr.size();
    for (int i = 0; i < n-1; ++i) {
        for (int j = 0; j < n-i-1; ++j) {
            if (arr[j] > arr[j+1]) {
                swap(arr[j], arr[j+1]);
            }
        }
    }
}

// Parallel Bubble Sort
void bubbleSortParallel(vector<int>& arr) {
    int n = arr.size();
    int i, j;
    #pragma omp parallel for private(i, j) shared(arr)
    for (i = 0; i < n-1; ++i) {
        for (j = 0; j < n-i-1; ++j) {
            if (arr[j] > arr[j+1]) {
                swap(arr[j], arr[j+1]);
            }
        }
    }
}

// Function to generate random array
vector<int> generateRandomArray(int size) {
    vector<int> arr(size);
    srand(time(0));
    for (int i = 0; i < size; ++i) {
        arr[i] = rand() % 1000; // Generating random numbers between 0 and 999
    }
    return arr;
}

// Function to measure time taken for sorting
void measurePerformance(vector<int>& arr, void (*sortFunc)(vector<int>&), const string& label) {
    auto start = high_resolution_clock::now();
    sortFunc(arr);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start).count();
    cout << label << " Sort Time: " << duration << " milliseconds" << endl;

    // Print sorted array
    cout << label << " Sorted Array: ";
    for (int i = 0; i < arr.size(); ++i) {
        cout << arr[i] << " ";
    }
    cout << endl;
}

int main() {
    int size = 10000; // Size of the array
    vector<int> arr = generateRandomArray(size);

    // Print generated array
    cout << "Generated Array: ";
    for (int i = 0; i < arr.size(); ++i) {
        cout << arr[i] << " ";
    }
    cout << endl;

    // Sequential Bubble Sort
    measurePerformance(arr, bubbleSortSequential, "Sequential");

    // Parallel Bubble Sort
    measurePerformance(arr, bubbleSortParallel, "Parallel");

    return 0;
}


/*
g++ -o graph bubblesort.cpp -fopenmp
./graph
*/