/**
* Copyright © 2018-2020 InAccel
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*   http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

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
