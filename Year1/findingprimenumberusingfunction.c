#include<stdio.h>
int isPrime(int number) {
  int k;
  for (k = 2; k <= number / 2; k++) {
    if (number % k == 0) {
      return 0;
    }
  }
  return 1;
}

void printPrime(int lowervalue, int uppervalue) {
  while (lowervalue <= uppervalue) {
    // Print if the current number is prime.
    if (isPrime(lowervalue)) {
      printf("%d",lowervalue);
    }
    lowervalue++;
  }
  return 0;
}