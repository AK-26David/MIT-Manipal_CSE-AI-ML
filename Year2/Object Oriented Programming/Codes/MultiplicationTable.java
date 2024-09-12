public class MultiplicationTable {
    public static void main(String[] args) {
        int m = 10;
        int n = 10;
        int[][] multiply = new int[m][n];

        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                multiply[i - 1][j - 1] = i * j;
            }
        }

        for (int[] row : multiply) {
            for (int cell : row) {
                System.out.print(cell + "\t");
            }
            System.out.println();
        }
    }
}
