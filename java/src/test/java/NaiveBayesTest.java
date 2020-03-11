import com.inaccel.ml.NaiveBayes;

public class NaiveBayesTest {
	public static void main(String[] args) {
		if (args.length != 0) System.out.println("Usage: java -cp <bla bla> " + args[0]);

		NaiveBayes nb = new NaiveBayes(26, 784);

		String home = System.getenv("HOME");

		nb.train(home + "/data/letters_csv_train.dat", 124800);

		float epsilon = 0.05f;

		nb.predict(epsilon);
	}
}
