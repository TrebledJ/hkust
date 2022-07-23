// Your Name: redacted
// Your Student ID: redacted
// Your email address: redacted
// Your COMP3711 lecture: L1

#include <algorithm> // std::min
#include <array>
#include <iostream>
#include <vector>
using namespace std;

bool setTrue(int a, int b, int c, int d, int e);
bool setFalse(int a, int b, int c, int d, int e);
bool report(int a, int b, int c, int d, int e);

void print(int x[]) {
  for (int i = 0; i < M; i++) {
    cout << " " << x[i];
  }
  cout << endl;
}

bool checkNext(int a, int b, int c, int d, int e) {
  // Loop through elements in Next(a, b, c, d, e).

  // For each row, we'll slice off a column (only squares on/above the current
  // row) and test if it is a losing position. Once we're finished with a row,
  // we restart from a clean slate.
  for (int row = 0; row < M; row++) {
    // Pack our little friends into an array. Note that this array is refreshed
    // each time we finish a row. We'll modify `x` directly to generate `x'`.
    int x[] = {a, b, c, d, e};

    // Our bounds. Start slicing from the right. End at the left.
    int start = x[row] - 1;

    // For the bottom row, only try up to column 1. (0, ..., 0) isn't considered
    // valid.
    int end = (row == 0 ? 1 : 0);

    // Let's slice!!!
    for (int i = start; i >= end; i--) {
      for (int r = row; r < M; r++) {
        // Chop off chocolates in rows above as well.
        if (x[r] > i) {
          x[r] = i;
        } else {
          // This row and the rows above all have x[i] <= check,
          // so we don't need to chop off these guys.
          break;
        }
      }

      if (!report(x[0], x[1], x[2], x[3], x[4])) {
        // Losing position found! :D
        return true;
      }
    }
  }

  // No losing position found... >.<
  return false;
}

void create(int n) {
  setFalse(1, 0, 0, 0, 0);

  // Construct table bottom-up. Iterate through all valid positions `x` and
  // check legal moves to `x'`.
  for (int a = 0; a <= n; a++)
    for (int b = 0; b <= a; b++)
      for (int c = 0; c <= b; c++)
        for (int d = 0; d <= c; d++)
          for (int e = 0; e <= d; e++) {
            bool win = checkNext(a, b, c, d, e);
            (win ? setTrue : setFalse)(a, b, c, d, e);
          }
}
