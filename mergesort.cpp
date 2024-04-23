#include <iostream>
#include <vector>
#include <chrono>
#include <omp.h>

using namespace std;
using namespace std::chrono;

// MERGE SORT
// 1) Divide (si, ei, mi)
// 2) Recursion (si to mi, mi+1 to ei)
// 3) Finally merge using temp array and then copy temp into original array

void merge(vector<int>& arr, int si, int ei, int mi) {
    // Initialize iterators & temp array
    vector<int> temp(ei - si + 1);
    int i = si;
    int j = mi + 1;
    int k = 0; // for temp array

    // 1) For left and right array
    while (i <= mi && j <= ei) {
        if (arr[i] < arr[j]) {
            temp[k] = arr[i];
            ++k;
            ++i;
        } else {
            temp[k] = arr[j];
            ++k;
            ++j;
        }
    }

    // 2) If right array is printed and left array remains then
    while (i <= mi) {
        temp[k] = arr[i];
        ++k;
        ++i;
    }

    // 3) If left array is printed and right array remains then
    while (j <= ei) {
        temp[k] = arr[j];
        ++k;
        ++j;
    }

    // Copying temp array to original array
    for (i = 0, j = si; i < temp.size(); ++i, ++j) {
        arr[j] = temp[i]; // Original array ka index si se start hoga, temp ka 0 se hi hoga.
    }
}

void mergeSortSequential(vector<int>& arr, int si, int ei) {
    // Base Case
    if (si >= ei) {
        return; // Since a single element is already sorted
    }

    // Initialize variable mi
    int mi = (si + ei) / 2;

    // Recursion
    mergeSortSequential(arr, si, mi);
    mergeSortSequential(arr, mi + 1, ei);

    // Merge
    merge(arr, si, ei, mi);
}

void mergeSortParallel(vector<int>& arr, int si, int ei) {
    // Base Case
    if (si >= ei) {
        return; // Since a single element is already sorted
    }

    // Initialize variable mi
    int mi = (si + ei) / 2;

    // Recursion with OpenMP tasks
    #pragma omp task
    mergeSortParallel(arr, si, mi);

    #pragma omp task
    mergeSortParallel(arr, mi + 1, ei);

    #pragma omp taskwait
    merge(arr, si, ei, mi);
}

void mergeSortWrapper(vector<int>& arr, int si, int ei) {
    #pragma omp parallel
    {
        #pragma omp single nowait
        mergeSortParallel(arr, si, ei);
    }
}

void measurePerformance(vector<int>& arr, void (*sortFunc)(vector<int>&, int, int), const string& label) {
    auto start = high_resolution_clock::now();
    sortFunc(arr, 0, arr.size() - 1);
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
    int size = 1000; // Size of the array
    vector<int> arr(size);

    srand(time(0));
    for (int i = 0; i < size; ++i) {
        arr[i] = rand() % 1000; // Generating random numbers between 0 and 999
    }

    // Print generated array
    cout << "Generated Array: ";
    for (int i = 0; i < arr.size(); ++i) {
        cout << arr[i] << " ";
    }
    cout << endl;

    // Sequential Merge Sort
    measurePerformance(arr, mergeSortSequential, "Sequential");

    // Parallel Merge Sort
    measurePerformance(arr, mergeSortWrapper, "Parallel");

    return 0;
}
